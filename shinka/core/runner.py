import json
import shutil
import sys
import signal
import uuid
import time
import logging
import yaml
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from rich.logging import RichHandler
from rich.table import Table
from rich.console import Console
import rich.box
from typing import List, Optional, Union, cast
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from subprocess import Popen
from shinka.launch import JobScheduler, JobConfig, ProcessWithLogging
from shinka.database import ProgramDatabase, DatabaseConfig, Program
from shinka.llm import (
    LLMClient,
    extract_between,
    EmbeddingClient,
    BanditBase,
    AsymmetricUCB,
    configure_pool,
)
from shinka.edit import (
    apply_diff_patch,
    apply_full_patch,
    summarize_diff,
    redact_immutable,
)
from shinka.core.sampler import PromptSampler
from shinka.core.summarizer import MetaSummarizer
from shinka.core.novelty_judge import NoveltyJudge
from shinka.logo import print_gradient_logo

FOLDER_PREFIX = "gen"


@dataclass
class EvolutionConfig:
    task_sys_msg: Optional[str] = None
    patch_types: List[str] = field(default_factory=lambda: ["diff"])
    patch_type_probs: List[float] = field(default_factory=lambda: [1.0])
    num_generations: int = 10
    max_parallel_jobs: int = 2
    max_patch_resamples: int = 3
    max_patch_attempts: int = 5
    job_type: str = "local"
    language: str = "python"
    llm_models: List[str] = field(default_factory=lambda: ["azure-gpt-4.1-mini"])
    llm_dynamic_selection: Optional[Union[str, BanditBase]] = None
    llm_dynamic_selection_kwargs: dict = field(default_factory=lambda: {})
    llm_kwargs: dict = field(default_factory=lambda: {})
    meta_rec_interval: Optional[int] = None
    meta_llm_models: Optional[List[str]] = None
    meta_llm_kwargs: dict = field(default_factory=lambda: {})
    meta_max_recommendations: int = 5
    embedding_model: Optional[str] = None
    init_program_path: Optional[str] = "initial.py"
    results_dir: Optional[str] = None
    max_novelty_attempts: int = 3
    code_embed_sim_threshold: float = 1.0
    novelty_llm_models: Optional[List[str]] = None
    novelty_llm_kwargs: dict = field(default_factory=lambda: {})
    use_text_feedback: bool = False


@dataclass
class RunningJob:
    """Represents a running job in the queue."""

    job_id: Union[str, Popen, ProcessWithLogging]
    exec_fname: str
    results_dir: str
    start_time: float
    generation: int
    parent_id: Optional[str]
    archive_insp_ids: List[str]
    top_k_insp_ids: List[str]
    code_diff: Optional[str]
    meta_patch_data: Optional[dict]
    code_embedding: List[float] = field(default_factory=list)
    embed_cost: float = 0.0
    novelty_cost: float = 0.0


# Set up logging
logger = logging.getLogger(__name__)


class EvolutionRunner:
    def __init__(
        self,
        evo_config: EvolutionConfig,
        job_config: JobConfig,
        db_config: DatabaseConfig,
        verbose: bool = True,
    ):
        self.evo_config = evo_config
        self.job_config = job_config
        self.db_config = db_config
        self.verbose = verbose

        # Initialize centralized LLM pool FIRST - before any LLM clients
        # This ensures all LLM calls are throttled by max_parallel_jobs
        self.llm_pool = configure_pool(max_concurrent=evo_config.max_parallel_jobs)

        print_gradient_logo((255, 0, 0), (255, 255, 255))
        if evo_config.results_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_dir = f"results_{timestamp}"
        else:
            self.results_dir = Path(evo_config.results_dir)

        if self.verbose:
            # Create log file path in results directory
            log_filename = f"{self.results_dir}/evolution_run.log"
            Path(self.results_dir).mkdir(parents=True, exist_ok=True)

            # Set up logging with both console and file handlers
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                handlers=[
                    RichHandler(
                        show_time=False, show_level=False, show_path=False
                    ),  # Console output (clean)
                    logging.FileHandler(
                        log_filename, mode="a", encoding="utf-8"
                    ),  # File output (detailed)
                ],
            )

            # Also log the initial setup information
            logger.info("=" * 80)
            start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Evolution run started at {start_time}")
            logger.info(f"Results directory: {self.results_dir}")
            logger.info(f"Log file: {log_filename}")
            logger.info("=" * 80)

        # Check if we are resuming a run
        resuming_run = False
        db_path = Path(f"{self.results_dir}/{db_config.db_path}")
        if self.evo_config.results_dir is not None and db_path.exists():
            resuming_run = True

        # Initialize LLM selection strategy
        if evo_config.llm_dynamic_selection is None:
            self.llm_selection = None
        elif isinstance(evo_config.llm_dynamic_selection, BanditBase):
            self.llm_selection = evo_config.llm_dynamic_selection
        elif (evo_config.llm_dynamic_selection.lower() == "ucb") or (
            evo_config.llm_dynamic_selection.lower() == "ucb1"
        ):
            self.llm_selection = AsymmetricUCB(
                arm_names=evo_config.llm_models,
                **evo_config.llm_dynamic_selection_kwargs,
            )
        else:
            raise ValueError("Invalid llm_dynamic_selection")

        # Initialize database and scheduler
        db_config.db_path = str(db_path)
        embedding_model_to_use = (
            evo_config.embedding_model or "text-embedding-3-small"
        )
        self.db = ProgramDatabase(
            config=db_config, embedding_model=embedding_model_to_use
        )
        self.scheduler = JobScheduler(
            job_type=evo_config.job_type,
            config=job_config,  # type: ignore
            verbose=verbose,
        )

        self.llm = LLMClient(
            model_names=evo_config.llm_models,
            model_selection=self.llm_selection,
            **evo_config.llm_kwargs,
            verbose=verbose,
        )
        if evo_config.embedding_model is not None:
            self.embedding = EmbeddingClient(
                model_name=evo_config.embedding_model,
                verbose=verbose,
            )
        else:
            self.embedding = None

        if evo_config.meta_llm_models is not None:
            self.meta_llm = LLMClient(
                model_names=evo_config.meta_llm_models,
                **evo_config.meta_llm_kwargs,
                verbose=verbose,
            )
        else:
            self.meta_llm = None

        if evo_config.novelty_llm_models is not None:
            self.novelty_llm = LLMClient(
                model_names=evo_config.novelty_llm_models,
                **evo_config.novelty_llm_kwargs,
                verbose=verbose,
            )
        else:
            self.novelty_llm = None

        # Initialize PromptSampler for handling LLM code prompts
        self.prompt_sampler = PromptSampler(
            task_sys_msg=evo_config.task_sys_msg,
            language=evo_config.language,
            patch_types=evo_config.patch_types,
            patch_type_probs=evo_config.patch_type_probs,
            use_text_feedback=evo_config.use_text_feedback,
        )

        # Initialize MetaSummarizer for meta-recommendations
        self.meta_summarizer = MetaSummarizer(
            meta_llm_client=self.meta_llm,
            language=evo_config.language,
            use_text_feedback=evo_config.use_text_feedback,
            max_recommendations=evo_config.meta_max_recommendations,
        )

        # Initialize NoveltyJudge for novelty assessment
        self.novelty_judge = NoveltyJudge(
            novelty_llm_client=self.novelty_llm,
            language=evo_config.language,
            similarity_threshold=evo_config.code_embed_sim_threshold,
            max_novelty_attempts=evo_config.max_novelty_attempts,
        )

        # Initialize rich console for formatted output
        self.console = Console()

        if self.evo_config.language == "cuda":
            self.lang_ext = "cu"
        elif self.evo_config.language == "cpp":
            self.lang_ext = "cpp"
        elif self.evo_config.language == "python":
            self.lang_ext = "py"
        elif self.evo_config.language == "rust":
            self.lang_ext = "rs"
        elif self.evo_config.language == "swift":
            self.lang_ext = "swift"
        elif self.evo_config.language in ["json", "json5"]:
            self.lang_ext = "json"
        else:
            msg = f"Language {self.evo_config.language} not supported"
            raise ValueError(msg)

        # Queue for managing parallel jobs
        self.running_jobs: List[RunningJob] = []
        self.best_program_id: Optional[str] = None
        self.next_generation_to_submit = 0

        # Threading lock for database operations and shared state
        self._db_lock = threading.Lock()
        self._generation_lock = threading.Lock()
        self._jobs_lock = threading.Lock()  # Protects running_jobs list

        # Shutdown flag - signal handler sets this, main thread checks it
        self._shutdown_requested = threading.Event()

        # Thread pool for parallel LLM calls
        self._llm_executor = ThreadPoolExecutor(max_workers=evo_config.max_parallel_jobs)

        if resuming_run:
            self.completed_generations = self.db.last_iteration + 1
            self.next_generation_to_submit = self.completed_generations
            self.session_start_generation = self.completed_generations  # Track session start
            logger.info("=" * 80)
            logger.info("RESUMING PREVIOUS EVOLUTION RUN")
            logger.info("=" * 80)

            # Verify database integrity before proceeding
            try:
                self.db.cursor.execute("PRAGMA integrity_check")
                integrity_result = self.db.cursor.fetchone()
                if integrity_result[0] != "ok":
                    logger.error(f"DATABASE INTEGRITY CHECK FAILED: {integrity_result}")
                    logger.error("The database may be corrupted. Consider restoring from backup.")
                else:
                    logger.info("Database integrity check: OK")
            except Exception as e:
                logger.warning(f"Could not verify database integrity: {e}")

            logger.info(
                f"Resuming evolution from: {self.results_dir}\n"
                f"Found {self.completed_generations} "
                "previously completed generations."
            )
            logger.info("=" * 80)

            # Recover any orphaned results (scorer completed but not in database)
            # This can happen if process crashed after scorer finished but before
            # _process_completed_job added the program to the database
            orphaned_recovered = self._recover_orphaned_results()
            if orphaned_recovered > 0:
                logger.info(f"Recovered {orphaned_recovered} orphaned results from previous session")
                # Update completed generations count after recovery
                self._update_completed_generations()

            # Recover ghost generations (main.py exists but scorer never ran)
            # This resubmits scorer jobs - they'll be processed in the main loop
            # Note: We don't wait for these here; the main loop handles them
            ghost_count = self._count_ghost_generations()
            if ghost_count > 0:
                logger.info(f"Found {ghost_count} ghost generations to recover (will resubmit in main loop)")

            self._update_best_solution()
            # Restore meta memory state when resuming
            self._restore_meta_memory()
            # Restore LLM selection bandit state when resuming
            self._restore_llm_selection_state()
        else:
            self.completed_generations = 0
            self.session_start_generation = 0  # Track session start

        # Save experiment configuration to a YAML file
        self._save_experiment_config(evo_config, job_config, db_config)

    def _save_experiment_config(
        self,
        evo_config: EvolutionConfig,
        job_config: JobConfig,
        db_config: DatabaseConfig,
    ) -> None:
        """Save experiment configuration to a YAML file."""
        config_data = {
            "evolution_config": asdict(evo_config),
            "job_config": asdict(job_config),
            "database_config": asdict(db_config),
            "timestamp": datetime.now().isoformat(),
            "results_directory": str(self.results_dir),
        }

        config_path = Path(self.results_dir) / "experiment_config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with config_path.open("w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)

        logger.info(f"Experiment configuration saved to {config_path}")

    def run(self):
        """Run evolution with parallel job queue."""
        # Set up signal handler for clean shutdown on Ctrl+C
        # IMPORTANT: Signal handler only sets flag - main thread does cleanup work
        # This prevents deadlocks from acquiring locks in signal context
        def graceful_shutdown(signum, frame):
            logger.info("")
            logger.info("=" * 60)
            logger.info("INTERRUPT RECEIVED - Requesting graceful shutdown...")
            logger.info("=" * 60)
            self._shutdown_requested.set()

        signal.signal(signal.SIGINT, graceful_shutdown)
        signal.signal(signal.SIGTERM, graceful_shutdown)

        max_jobs = self.evo_config.max_parallel_jobs
        target_gens = self.evo_config.num_generations
        logger.info(
            f"Starting evolution with {max_jobs} parallel jobs, "
            f"target: {target_gens} generations"
        )

        # First, run generation 0 sequentially to populate the database
        if self.completed_generations == 0 and target_gens > 0:
            logger.info("Running generation 0 sequentially to initialize database...")
            self._run_generation_0()
            self.completed_generations = 1
            self.next_generation_to_submit = 1
            logger.info(f"Completed generation 0, total: 1/{target_gens}")

        # Now start parallel execution for remaining generations
        if self.completed_generations < target_gens:
            logger.info("Starting parallel execution for remaining generations...")

            # Periodic checkpoint tracking (CRITICAL: protects against SIGKILL)
            last_checkpoint_time = time.time()
            checkpoint_interval = 60  # seconds - save state every minute

            # Main loop: monitor jobs and submit new ones
            while True:
                with self._jobs_lock:
                    has_running_jobs = len(self.running_jobs) > 0
                if self.completed_generations >= target_gens and not has_running_jobs:
                    break
                # Check for shutdown request (set by signal handler)
                if self._shutdown_requested.is_set():
                    self._perform_graceful_shutdown()
                    return

                # Check for completed jobs
                completed_jobs = self._check_completed_jobs()

                # Process completed jobs
                if completed_jobs:
                    for job in completed_jobs:
                        self._process_completed_job(job)

                    # Update completed generations count
                    self._update_completed_generations()

                    if self.verbose:
                        session_progress = self.completed_generations - self.session_start_generation
                        logger.info(
                            f"Processed {len(completed_jobs)} jobs. "
                            f"Total generations: {self.completed_generations}/{target_gens} "
                            f"({session_progress} this session)"
                        )

                # Periodic checkpoint - protects against SIGKILL and hard crashes
                if time.time() - last_checkpoint_time > checkpoint_interval:
                    try:
                        self.db.save()
                        self._save_meta_memory()
                        last_checkpoint_time = time.time()
                        if self.verbose:
                            logger.debug("Periodic checkpoint saved")
                    except Exception as e:
                        logger.warning(f"Periodic checkpoint failed: {e}")

                # Check if we've completed all generations
                if self.completed_generations >= target_gens:
                    logger.info("All generations completed, exiting...")
                    break

                # Submit new jobs to fill the queue (parallel submission)
                with self._jobs_lock:
                    running_count = len(self.running_jobs)
                available_slots = max_jobs - running_count
                jobs_to_submit = min(
                    available_slots,
                    target_gens - self.next_generation_to_submit
                )

                if jobs_to_submit > 0:
                    # Submit multiple jobs in parallel using thread pool
                    futures = []
                    for _ in range(jobs_to_submit):
                        if self.next_generation_to_submit < target_gens:
                            future = self._llm_executor.submit(self._submit_new_job)
                            futures.append(future)

                    # Process completions while waiting for LLM submissions (non-blocking)
                    pending_futures = set(futures)
                    while pending_futures:
                        # Check for shutdown request in inner loop too
                        if self._shutdown_requested.is_set():
                            self._perform_graceful_shutdown()
                            return

                        # Recover any ghost generations (called frequently to minimize lost work)
                        ghost_recovered = self._recover_ghost_generations()
                        if ghost_recovered > 0:
                            logger.info(f"Recovered {ghost_recovered} ghost generations")

                        # Check for completed evaluation jobs
                        completed_jobs = self._check_completed_jobs()
                        if completed_jobs:
                            for job in completed_jobs:
                                self._process_completed_job(job)
                            self._update_completed_generations()
                            if self.verbose:
                                session_progress = self.completed_generations - self.session_start_generation
                                logger.info(
                                    f"Processed {len(completed_jobs)} jobs. "
                                    f"Total generations: {self.completed_generations}/{target_gens} "
                                    f"({session_progress} this session)"
                                )

                        # Non-blocking check of LLM submissions (0.5s timeout)
                        done, pending_futures = wait(pending_futures, timeout=0.5, return_when=FIRST_COMPLETED)
                        for future in done:
                            try:
                                future.result()
                            except Exception as e:
                                logger.error(f"Error in parallel job submission: {e}")
                                # Immediately try to recover ghost generations after errors
                                recovered = self._recover_ghost_generations()
                                if recovered > 0:
                                    logger.info(f"Recovered {recovered} ghost generations after error")
                else:
                    # No jobs to submit, just wait a bit
                    time.sleep(0.5)

            # All jobs are now handled by the main loop above

        # Perform final meta summary for any remaining unprocessed programs
        best_program = self.db.get_best_program()
        self.meta_summarizer.perform_final_summary(str(self.results_dir), best_program)

        # Save final meta memory state
        self._save_meta_memory()

        self.db.print_summary()
        logger.info(f"Evolution completed! {self.completed_generations} generations")

        # Log LLM pool statistics
        pool_stats = self.llm_pool.get_stats()
        logger.info(
            f"LLM Pool Stats: {pool_stats['total_requests']} requests, "
            f"peak concurrent: {pool_stats['peak_concurrent']}/{pool_stats['max_concurrent']}, "
            f"total cost: ${pool_stats['total_cost']:.4f}"
        )

        logger.info("=" * 80)
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Evolution run ended at {end_time}")
        logger.info("=" * 80)

    def generate_initial_program(self):
        """Generate initial program with LLM, with retries."""
        llm_kwargs = self.llm.get_kwargs()

        sys_msg, user_msg = self.prompt_sampler.initial_program_prompt()
        msg_history = []
        total_costs = 0.0

        for attempt in range(self.evo_config.max_patch_attempts):
            response = self.llm.query(
                msg=user_msg,
                system_msg=sys_msg,
                llm_kwargs=llm_kwargs,
                msg_history=msg_history,
            )
            if response is None or response.content is None:
                if self.verbose:
                    logger.info(
                        f"  INITIAL PROGRAM ATTEMPT {attempt + 1}/"
                        f"{self.evo_config.max_patch_attempts} "
                        "FAILURE. Error: LLM response content was None."
                    )
                if attempt < self.evo_config.max_patch_attempts - 1:
                    user_msg = (
                        "The previous response was empty. Please try again "
                        "and provide the full code."
                    )
                    if response and response.new_msg_history:
                        msg_history = response.new_msg_history
                    continue
                else:
                    break

            total_costs += response.cost or 0
            initial_code = extract_between(
                response.content,
                f"```{self.evo_config.language}",
                "```",
                False,
            )

            if initial_code:
                patch_name = extract_between(
                    response.content, "<NAME>", "</NAME>", False
                )
                patch_description = extract_between(
                    response.content, "<DESCRIPTION>", "</DESCRIPTION>", False
                )
                if self.evo_config.language == "python":
                    comment_char = "#"
                else:
                    comment_char = "//"

                initial_code = (
                    f"{comment_char} EVOLVE-BLOCK-START\n"
                    f"{initial_code}\n"
                    f"{comment_char} EVOLVE-BLOCK-END\n"
                )

                if self.verbose:
                    logger.info(
                        f"  INITIAL PROGRAM ATTEMPT {attempt + 1}/"
                        f"{self.evo_config.max_patch_attempts} "
                        "SUCCESS."
                    )
                return initial_code, patch_name, patch_description, total_costs
            else:  # code extraction failed
                if self.verbose:
                    logger.info(
                        f"  INITIAL PROGRAM ATTEMPT {attempt + 1}/"
                        f"{self.evo_config.max_patch_attempts} "
                        "FAILURE. Error: Could not extract code from response."
                    )
                if attempt < self.evo_config.max_patch_attempts - 1:
                    user_msg = (
                        "Could not extract code from your last response. "
                        "Please make sure to enclose the code in "
                        "`<CODE>`...`</CODE>` tags."
                    )
                    msg_history = response.new_msg_history
                else:  # last attempt
                    break

        raise ValueError(
            "LLM failed to generate a valid initial program after "
            f"{self.evo_config.max_patch_attempts} attempts."
        )

    def _run_generation_0(self):
        """Setup and run generation 0 to initialize the database."""
        initial_dir = f"{self.results_dir}/{FOLDER_PREFIX}_0"
        Path(initial_dir).mkdir(parents=True, exist_ok=True)
        exec_fname = f"{initial_dir}/main.{self.lang_ext}"
        results_dir = f"{self.results_dir}/{FOLDER_PREFIX}_0/results"
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        api_costs = 0.0
        patch_name = "initial_program"
        patch_description = "Initial program from file."
        patch_type = "init"

        if self.evo_config.init_program_path:
            if self.verbose:
                logger.info(
                    f"Copying initial program from {self.evo_config.init_program_path}"
                )
            shutil.copy(self.evo_config.init_program_path, exec_fname)
        else:
            if self.verbose:
                logger.info(
                    "`init_program_path` not provided, "
                    "generating initial program with LLM..."
                )
            initial_code, patch_name, patch_description, api_costs = (
                self.generate_initial_program()
            )
            with open(exec_fname, "w", encoding="utf-8") as f:
                f.write(initial_code)

            if self.verbose:
                logger.info(f"Initial program generated and saved to {exec_fname}")

        # Run the evaluation synchronously
        results, rtime = self.scheduler.run(exec_fname, results_dir)

        code_embedding, e_cost = self.get_code_embedding(exec_fname)

        # Read the evaluated code for database insertion
        try:
            evaluated_code = Path(exec_fname).read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not read code for job {exec_fname}. Error: {e}")
            evaluated_code = ""

        correct_val = False
        metrics_val = {}
        stdout_log = ""
        stderr_log = ""
        if results:
            correct_val = results.get("correct", {}).get("correct", False)
            metrics_val = results.get("metrics", {})
            stdout_log = results.get("stdout_log", "")
            stderr_log = results.get("stderr_log", "")

        combined_score = metrics_val.get("combined_score", 0.0)
        public_metrics = metrics_val.get("public", {})
        private_metrics = metrics_val.get("private", {})
        text_feedback = metrics_val.get("text_feedback", "")

        # Add the program to the database
        db_program = Program(
            id=str(uuid.uuid4()),
            code=evaluated_code,
            language=self.evo_config.language,
            parent_id=None,
            generation=0,
            archive_inspiration_ids=[],
            top_k_inspiration_ids=[],
            code_diff=None,
            embedding=code_embedding,
            correct=correct_val,
            combined_score=combined_score,
            public_metrics=public_metrics,
            private_metrics=private_metrics,
            text_feedback=text_feedback,
            metadata={
                "compute_time": rtime,
                "api_costs": api_costs,
                "embed_cost": e_cost,
                "novelty_cost": 0.0,  # No novelty cost for generation 0
                "patch_type": patch_type,
                "patch_name": patch_name,
                "patch_description": patch_description,
                "stdout_log": stdout_log,
                "stderr_log": stderr_log,
            },
        )

        self.db.add(db_program, verbose=True)
        if self.llm_selection is not None:
            self.llm_selection.set_baseline_score(
                db_program.combined_score if correct_val else 0.0,
            )
        self.db.save()
        self._update_best_solution()

        # Add the evaluated program to meta memory tracking
        self.meta_summarizer.add_evaluated_program(db_program)

        # Check if we should update meta memory after adding this program
        if self.meta_summarizer.should_update_meta(self.evo_config.meta_rec_interval):
            logger.info(
                f"Updating meta memory after processing "
                f"{len(self.meta_summarizer.evaluated_since_last_meta)} programs..."
            )
            best_program = self.db.get_best_program()
            updated_recs, meta_cost = self.meta_summarizer.update_meta_memory(
                best_program
            )
            if updated_recs:
                # Write meta output file for generation 0
                self.meta_summarizer.write_meta_output(str(self.results_dir))
                # Store meta cost for tracking
                if meta_cost > 0:
                    logger.info(
                        f"Meta recommendation generation cost: ${meta_cost:.4f}"
                    )
                    # Add meta cost to this program's metadata (the one that triggered the update)
                    if db_program.metadata is None:
                        db_program.metadata = {}
                    db_program.metadata["meta_cost"] = meta_cost
                    # Update the program in the database with the new metadata
                    metadata_json = json.dumps(db_program.metadata)
                    self.db.cursor.execute(
                        "UPDATE programs SET metadata = ? WHERE id = ?",
                        (metadata_json, db_program.id),
                    )
                    self.db.conn.commit()

        # Save meta memory state after each job completion
        self._save_meta_memory()

    def _update_completed_generations(self):
        """
        Update the count of completed generations from the database.
        Uses the maximum generation number (0-indexed), not contiguous count.
        This gives accurate progress reporting even with gaps in generation numbers.
        """
        with self._db_lock:
            last_gen = self.db.last_iteration
            if last_gen == -1:
                self.completed_generations = 0
            else:
                # Use max generation + 1 as the count (0-indexed)
                self.completed_generations = last_gen + 1

    def _submit_new_job(self):
        """Submit a new job to the queue (thread-safe)."""
        # Thread-safe generation counter increment
        with self._generation_lock:
            current_gen = self.next_generation_to_submit
            if current_gen >= self.evo_config.num_generations:
                return
            self.next_generation_to_submit += 1

        exec_fname = (
            f"{self.results_dir}/{FOLDER_PREFIX}_{current_gen}/main.{self.lang_ext}"
        )
        results_dir = f"{self.results_dir}/{FOLDER_PREFIX}_{current_gen}/results"
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        # Get current meta-recommendations for this job
        meta_recs, meta_summary, meta_scratch = self.meta_summarizer.get_current()

        # Sample parent and inspiration programs
        if current_gen == 0:
            parent_id = None
            archive_insp_ids = []
            top_k_insp_ids = []
            code_diff = None
            meta_patch_data = {}
            # Defensive initialization - gen 0 should be handled by _run_generation_0()
            # but initialize these for safety in case of unexpected code paths
            code_embedding = None
            embed_cost = 0.0
            novelty_cost = 0.0
            # Initial program already copied in setup_initial_program
        else:
            api_costs = 0
            embed_cost = 0
            novelty_cost = 0.0
            novelty_checks_performed = 0
            # Loop over novelty attempts
            for nov_attempt in range(self.evo_config.max_novelty_attempts):
                # Loop over patch resamples - including parents
                for resample in range(self.evo_config.max_patch_resamples):
                    # Thread-safe database sampling
                    with self._db_lock:
                        (
                            parent_program,
                            archive_programs,
                            top_k_programs,
                        ) = self.db.sample(
                            target_generation=current_gen,
                            novelty_attempt=nov_attempt + 1,
                            max_novelty_attempts=self.evo_config.max_novelty_attempts,
                            resample_attempt=resample + 1,
                            max_resample_attempts=self.evo_config.max_patch_resamples,
                        )
                    archive_insp_ids = [p.id for p in archive_programs]
                    top_k_insp_ids = [p.id for p in top_k_programs]
                    parent_id = parent_program.id
                    # Run patch (until success with max attempts)
                    code_diff, meta_patch_data, num_applied_attempt = self.run_patch(
                        parent_program,
                        archive_programs,
                        top_k_programs,
                        current_gen,
                        novelty_attempt=nov_attempt + 1,
                        resample_attempt=resample + 1,
                    )
                    api_costs += meta_patch_data["api_costs"]
                    if (
                        meta_patch_data["error_attempt"] is None
                        and num_applied_attempt > 0
                    ):
                        meta_patch_data["api_costs"] = api_costs
                        break

                # Get the code embedding for the evaluated code
                code_embedding, e_cost = self.get_code_embedding(exec_fname)
                embed_cost += e_cost

                if not code_embedding:
                    self.novelty_judge.log_novelty_skip_message("no embedding")
                    break

                # Use NoveltyJudge for novelty assessment with rejection sampling
                # Protect all DB accesses during novelty assessment with lock
                with self._db_lock:
                    should_check = self.novelty_judge.should_check_novelty(
                        code_embedding, current_gen, parent_program, self.db
                    )
                    if should_check:
                        should_accept, novelty_metadata = (
                            self.novelty_judge.assess_novelty_with_rejection_sampling(
                                exec_fname, code_embedding, parent_program, self.db
                            )
                        )
                    else:
                        should_accept = True  # Skip novelty check
                        novelty_metadata = {}
                        if not self.db.island_manager or not hasattr(
                            self.db.island_manager, "are_all_islands_initialized"
                        ):
                            self.novelty_judge.log_novelty_skip_message("no island manager")
                        elif not self.db.island_manager.are_all_islands_initialized():
                            self.novelty_judge.log_novelty_skip_message(
                                "not all islands initialized yet"
                            )

                # Update costs and metadata from novelty assessment (outside lock)
                if novelty_metadata:
                    novelty_cost += novelty_metadata.get("novelty_total_cost", 0.0)
                    novelty_checks_performed = novelty_metadata.get(
                        "novelty_checks_performed", 0
                    )
                    novelty_explanation = novelty_metadata.get(
                        "novelty_explanation", ""
                    )

                if should_accept:
                    break
                # If not accepted, continue to next attempt (rejection sampling)

        # Add meta-recommendations/summary/scratchpad to meta_patch_data
        if meta_recs is not None:
            meta_patch_data["meta_recommendations"] = meta_recs
            meta_patch_data["meta_summary"] = meta_summary
            meta_patch_data["meta_scratch_pad"] = meta_scratch

        # Add novelty check information to meta_patch_data if any checks were performed
        if current_gen > 0 and novelty_checks_performed > 0:
            meta_patch_data["novelty_checks_performed"] = novelty_checks_performed
            meta_patch_data["novelty_cost"] = novelty_cost
            meta_patch_data["novelty_explanation"] = novelty_explanation

        # Submit the job asynchronously
        job_id = self.scheduler.submit_async(exec_fname, results_dir)

        # Add to running jobs queue (thread-safe)
        running_job = RunningJob(
            job_id=job_id,
            exec_fname=exec_fname,
            results_dir=results_dir,
            start_time=time.time(),
            generation=current_gen,
            parent_id=parent_id,
            archive_insp_ids=archive_insp_ids,
            top_k_insp_ids=top_k_insp_ids,
            code_diff=code_diff,
            meta_patch_data=meta_patch_data,
            code_embedding=code_embedding,
            embed_cost=embed_cost,
            novelty_cost=novelty_cost,
        )
        with self._jobs_lock:
            self.running_jobs.append(running_job)
            queue_size = len(self.running_jobs)

        if self.verbose:
            logger.info(
                f"Submitted job for generation {current_gen}, "
                f"queue size: {queue_size}"
            )

    def _perform_graceful_shutdown(self):
        """
        Perform graceful shutdown - called from main thread when shutdown flag is set.

        This processes completed jobs and saves state before exiting.
        MUST be called from main thread (not signal handler) to avoid deadlocks.
        """
        logger.info("Performing graceful shutdown...")

        # Step 1: Shut down thread pool to prevent new submissions
        try:
            logger.info("Shutting down thread pool...")
            self._llm_executor.shutdown(wait=False, cancel_futures=True)
        except Exception as e:
            logger.warning(f"Thread pool shutdown error (non-fatal): {e}")

        # Step 2: Process any completed jobs before saving
        try:
            completed_jobs = self._check_completed_jobs()
            if completed_jobs:
                logger.info(f"Processing {len(completed_jobs)} completed jobs before exit...")
                for job in completed_jobs:
                    try:
                        self._process_completed_job(job)
                        logger.info(f"  Saved job for generation {job.generation}")
                    except Exception as e:
                        logger.error(f"  Failed to save job {job.generation}: {e}")
        except Exception as e:
            logger.error(f"Error processing completed jobs: {e}")

        # Step 3: Save state with retries (CRITICAL)
        max_retries = 3
        for retry in range(max_retries):
            try:
                self._save_meta_memory()  # Also saves LLM selection state
                self.db.save()
                logger.info("State saved successfully. Safe to exit.")
                logger.info(f"Resume with: --resume {self.results_dir}")
                break
            except Exception as e:
                if retry < max_retries - 1:
                    logger.warning(f"State save failed (attempt {retry + 1}/{max_retries}): {e}")
                    time.sleep(0.5)
                else:
                    logger.critical(f"FAILED TO SAVE STATE AFTER {max_retries} RETRIES: {e}")
                    logger.critical("Data may be lost! Check database integrity on resume.")

        logger.info("=" * 60)

    def _count_ghost_generations(self) -> int:
        """
        Count ghost generations (main.py exists but scorer never ran) without recovering them.
        Used on resume to report how many ghosts will be recovered in the main loop.
        """
        count = 0
        with self._jobs_lock:
            running_gens = {job.generation for job in self.running_jobs}

        # Scan all gen_* directories to count ghosts
        results_path = Path(self.results_dir)
        for gen_dir in results_path.glob(f"{FOLDER_PREFIX}_*"):
            try:
                gen_idx = int(gen_dir.name.split("_")[1])
            except (IndexError, ValueError):
                continue

            if gen_idx in running_gens:
                continue

            main_file = gen_dir / f"main.{self.lang_ext}"
            metrics_file = gen_dir / "results" / "metrics.json"
            job_log = gen_dir / "results" / "job_log.out"

            # Ghost: main.py exists, no metrics.json, no job log (scorer never ran)
            if main_file.exists() and not metrics_file.exists() and not job_log.exists():
                count += 1

        return count

    def _recover_ghost_generations(self) -> int:
        """
        Recover 'ghost generations' - generations where main.py exists but scorer never ran.

        This can happen when job submission fails due to database concurrency errors
        (e.g., 'Recursive use of cursors not allowed').

        Returns:
            Number of ghost generations recovered
        """
        recovered = 0
        with self._jobs_lock:
            running_gens = {job.generation for job in self.running_jobs}

        # Debug: log scanning range
        if self.verbose:
            logger.debug(
                f"Ghost recovery: scanning gens 0-{self.next_generation_to_submit-1}, "
                f"running: {sorted(running_gens)[:10]}..."
            )

        # Scan for generations with main.py but no metrics.json
        for gen_idx in range(self.next_generation_to_submit):
            if gen_idx in running_gens:
                continue  # Already in queue

            gen_dir = Path(self.results_dir) / f"{FOLDER_PREFIX}_{gen_idx}"
            main_file = gen_dir / f"main.{self.lang_ext}"
            results_dir = gen_dir / "results"
            metrics_file = results_dir / "metrics.json"

            # Check if this is a ghost generation
            if main_file.exists() and not metrics_file.exists():
                # Check if there's already a scorer running (job_log files exist and are recent)
                job_log = results_dir / "job_log.out"
                if job_log.exists():
                    # Check if job log is recent (within 5 minutes) - scorer might still be running
                    try:
                        log_age = time.time() - job_log.stat().st_mtime
                        if log_age < 300:  # Less than 5 minutes old
                            logger.debug(
                                f"Skipping ghost gen {gen_idx}: job_log is recent "
                                f"({log_age:.0f}s old), scorer may still be running"
                            )
                            continue
                        # Log file exists but is old - scorer failed or was killed
                        logger.debug(f"Ghost gen {gen_idx}: job_log is stale ({log_age:.0f}s old)")
                    except OSError:
                        pass  # Stat failed, proceed with recovery

                # This is a ghost generation - resubmit scorer
                logger.warning(
                    f"Recovering ghost generation {gen_idx}: "
                    f"main.py exists but scorer never ran"
                )

                try:
                    # Ensure results directory exists
                    results_dir.mkdir(parents=True, exist_ok=True)

                    # Try to recover parent_id from LLM response files
                    parent_id = None
                    for attempt in range(1, 6):  # Check up to 5 attempts
                        llm_response_file = gen_dir / f"llm_response_attempt_{attempt}.json"
                        if llm_response_file.exists():
                            try:
                                with open(llm_response_file, 'r') as f:
                                    llm_data = json.load(f)
                                    parent_id = llm_data.get("parent_id")
                                    if parent_id:
                                        logger.info(f"Recovered parent_id {parent_id} for ghost gen {gen_idx}")
                                        break
                            except Exception as e:
                                logger.debug(f"Could not read LLM response file: {e}")

                    # Submit the scorer job
                    job_id = self.scheduler.submit_async(str(main_file), str(results_dir))

                    # Add to running jobs with recovered metadata where possible
                    running_job = RunningJob(
                        job_id=job_id,
                        exec_fname=str(main_file),
                        results_dir=str(results_dir),
                        start_time=time.time(),
                        generation=gen_idx,
                        parent_id=parent_id,  # Recovered from LLM response if available
                        archive_insp_ids=[],
                        top_k_insp_ids=[],
                        code_diff=None,
                        meta_patch_data={"recovered_ghost": True},
                        code_embedding=None,
                        embed_cost=0.0,
                        novelty_cost=0.0,
                    )
                    with self._jobs_lock:
                        self.running_jobs.append(running_job)
                        queue_size = len(self.running_jobs)
                    recovered += 1

                    logger.info(
                        f"Resubmitted scorer for ghost generation {gen_idx}, "
                        f"queue size: {queue_size}"
                    )
                except Exception as e:
                    logger.error(f"Failed to recover ghost generation {gen_idx}: {e}")

        return recovered

    def _recover_orphaned_results(self) -> int:
        """
        Recover 'orphaned results' - generations where scorer completed (metrics.json exists)
        but process crashed before _process_completed_job added the program to the database.

        This complements _recover_ghost_generations which handles missing metrics.json.
        Scans ALL gen_* directories, not just up to next_generation_to_submit,
        to catch orphans from any previous session.

        Returns:
            Number of orphaned results recovered
        """
        recovered = 0
        with self._jobs_lock:
            running_gens = {job.generation for job in self.running_jobs}

        # Scan ALL gen_* directories, not just up to next_generation_to_submit
        # This catches orphans from previous sessions with higher generation numbers
        results_path = Path(self.results_dir)
        for gen_dir in results_path.glob(f"{FOLDER_PREFIX}_*"):
            try:
                gen_idx = int(gen_dir.name.split("_")[1])
            except (IndexError, ValueError):
                continue  # Invalid directory name

            if gen_idx == 0:
                continue  # Skip gen 0 (handled separately)
            if gen_idx in running_gens:
                continue  # Already being processed

            main_file = gen_dir / f"main.{self.lang_ext}"
            results_dir_path = gen_dir / "results"
            metrics_file = results_dir_path / "metrics.json"
            correct_file = results_dir_path / "correct.json"

            # Only process if metrics.json exists (scorer completed)
            if not metrics_file.exists():
                continue  # Ghost recovery handles this case

            # Check if this generation is already in the database
            # NOTE: We check again inside the lock before inserting to prevent race conditions
            with self._db_lock:
                self.db.cursor.execute(
                    "SELECT COUNT(*) FROM programs WHERE generation = ?",
                    (gen_idx,)
                )
                count = self.db.cursor.fetchone()[0]
                if count > 0:
                    continue  # Already in database

            # This is an orphaned result - scorer finished but not in database
            logger.warning(
                f"Recovering orphaned result for gen {gen_idx}: "
                f"metrics.json exists but not in database"
            )

            try:
                # Read metrics directly from files (don't rely on scheduler)
                metrics_val = {}
                correct_val = False
                text_feedback = ""

                try:
                    with open(metrics_file, 'r') as f:
                        metrics_val = json.load(f)
                    logger.info(f"Read metrics for gen {gen_idx}: score={metrics_val.get('combined_score')}")
                except Exception as e:
                    logger.warning(f"Could not read metrics.json for gen {gen_idx}: {e}")

                try:
                    with open(correct_file, 'r') as f:
                        correct_data = json.load(f)
                        correct_val = correct_data.get("correct", False)
                except Exception as e:
                    logger.warning(f"Could not read correct.json for gen {gen_idx}: {e}")

                # Try to recover parent_id from LLM response files
                parent_id = None
                code_diff = None
                for attempt in range(1, 6):  # Check up to 5 attempts
                    llm_response_file = gen_dir / f"llm_response_attempt_{attempt}.json"
                    if llm_response_file.exists():
                        try:
                            with open(llm_response_file, 'r') as f:
                                llm_data = json.load(f)
                                parent_id = llm_data.get("parent_id")
                                if parent_id:
                                    logger.info(f"Recovered parent_id {parent_id} for orphaned gen {gen_idx}")
                                    break
                        except Exception as e:
                            logger.debug(f"Could not read LLM response file: {e}")

                # Read the code diff if available
                diff_file = gen_dir / "edit.diff"
                if diff_file.exists():
                    try:
                        code_diff = diff_file.read_text(encoding="utf-8")
                    except Exception:
                        pass

                # Read the code
                evaluated_code = ""
                try:
                    evaluated_code = main_file.read_text(encoding="utf-8")
                except Exception as e:
                    logger.warning(f"Could not read main.py for gen {gen_idx}: {e}")

                # Extract metrics
                combined_score = metrics_val.get("combined_score", 0.0)
                public_metrics = metrics_val.get("public", {})
                private_metrics = metrics_val.get("private", {})
                text_feedback = metrics_val.get("text_feedback", "")

                # Create program and add to database directly
                db_program = Program(
                    id=str(uuid.uuid4()),
                    code=evaluated_code,
                    language=self.evo_config.language,
                    parent_id=parent_id,
                    generation=gen_idx,
                    archive_inspiration_ids=[],
                    top_k_inspiration_ids=[],
                    code_diff=code_diff,
                    embedding=None,
                    correct=correct_val,
                    combined_score=combined_score,
                    public_metrics=public_metrics,
                    private_metrics=private_metrics,
                    text_feedback=text_feedback,
                    metadata={"recovered_orphan": True},
                )

                with self._db_lock:
                    # Double-check inside lock to prevent race condition
                    # Another thread may have added this generation since our first check
                    self.db.cursor.execute(
                        "SELECT COUNT(*) FROM programs WHERE generation = ?",
                        (gen_idx,)
                    )
                    if self.db.cursor.fetchone()[0] > 0:
                        logger.debug(f"Gen {gen_idx} already added by another thread, skipping")
                        continue  # Another thread beat us to it

                    self.db.add(db_program, verbose=True)
                    self.db.save()

                recovered += 1
                logger.info(
                    f"Recovered orphaned result for generation {gen_idx}: "
                    f"score={combined_score}, correct={correct_val}"
                )

            except Exception as e:
                logger.error(f"Failed to recover orphaned result for gen {gen_idx}: {e}")

        return recovered

    def _check_completed_jobs(self) -> List[RunningJob]:
        """Check for completed jobs and return them (thread-safe)."""
        completed = []
        still_running = []

        with self._jobs_lock:
            jobs_snapshot = list(self.running_jobs)

        for job in jobs_snapshot:
            is_running = self.scheduler.check_job_status(job)
            if not is_running:
                # Job completed
                if self.verbose:
                    logger.info(f"Job {job.job_id} completed!")
                completed.append(job)
            else:
                # Job still running
                still_running.append(job)

        with self._jobs_lock:
            # Only remove completed jobs from the list - new jobs may have been added
            completed_gens = {job.generation for job in completed}
            self.running_jobs = [j for j in self.running_jobs if j.generation not in completed_gens]

        return completed

    def _process_completed_job(self, job: RunningJob):
        """Process a completed job and add results to database."""
        end_time = time.time()
        rtime = end_time - job.start_time

        # Get job results
        results = self.scheduler.get_job_results(job.job_id, job.results_dir)

        # Read the evaluated code
        try:
            evaluated_code = Path(job.exec_fname).read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not read code for job {job.job_id}. Error: {e}")
            evaluated_code = ""

        # Use pre-computed embedding and novelty costs
        code_embedding = job.code_embedding
        e_cost = job.embed_cost if job.embed_cost is not None else 0.0
        n_cost = job.novelty_cost if job.novelty_cost is not None else 0.0
        if self.verbose:
            logger.debug(
                f"=> Using pre-computed embedding for job {job.job_id}, "
                f"embed cost: {e_cost:.4f}, novelty cost: {n_cost:.4f}"
            )

        correct_val = False
        metrics_val = {}
        stdout_log = ""
        stderr_log = ""
        if results:
            correct_val = results.get("correct", {}).get("correct", False)
            metrics_val = results.get("metrics", {})
            stdout_log = results.get("stdout_log", "")
            stderr_log = results.get("stderr_log", "")

        combined_score = metrics_val.get("combined_score", 0.0)
        public_metrics = metrics_val.get("public", {})
        private_metrics = metrics_val.get("private", {})
        text_feedback = metrics_val.get("text_feedback", "")

        # Add the program to the database
        db_program = Program(
            id=str(uuid.uuid4()),
            code=evaluated_code,
            language=self.evo_config.language,
            parent_id=job.parent_id,
            generation=job.generation,
            archive_inspiration_ids=job.archive_insp_ids,
            top_k_inspiration_ids=job.top_k_insp_ids,
            code_diff=job.code_diff,
            embedding=code_embedding,
            correct=correct_val,
            combined_score=combined_score,
            public_metrics=public_metrics,
            private_metrics=private_metrics,
            text_feedback=text_feedback,
            metadata={
                "compute_time": rtime,
                **(job.meta_patch_data or {}),
                "embed_cost": e_cost,
                "novelty_cost": n_cost,
                "stdout_log": stdout_log,
                "stderr_log": stderr_log,
            },
        )

        # Protect ALL database operations with lock to prevent
        # "Recursive use of cursors" error when worker threads run concurrently
        with self._db_lock:
            self.db.add(db_program, verbose=True)

            # Add the evaluated program to meta memory tracking
            self.meta_summarizer.add_evaluated_program(db_program)

            # Check if we should update meta memory after adding this program
            if self.meta_summarizer.should_update_meta(self.evo_config.meta_rec_interval):
                logger.info(
                    f"Updating meta memory after processing "
                    f"{len(self.meta_summarizer.evaluated_since_last_meta)} programs..."
                )
                best_program = self.db.get_best_program()
                updated_recs, meta_cost = self.meta_summarizer.update_meta_memory(
                    best_program
                )
                if updated_recs:
                    # Write meta output file using accumulated program count
                    self.meta_summarizer.write_meta_output(str(self.results_dir))
                    # Store meta cost for tracking
                    if meta_cost > 0:
                        logger.info(
                            f"Meta recommendation generation cost: ${meta_cost:.4f}"
                        )
                        # Add meta cost to this program's metadata (the one that triggered the update)
                        if db_program.metadata is None:
                            db_program.metadata = {}
                        db_program.metadata["meta_cost"] = meta_cost
                        # Update the program in the database with the new metadata
                        metadata_json = json.dumps(db_program.metadata)
                        self.db.cursor.execute(
                            "UPDATE programs SET metadata = ? WHERE id = ?",
                            (metadata_json, db_program.id),
                        )
                        self.db.conn.commit()

            if self.llm_selection is not None:
                if "model_name" not in db_program.metadata:
                    logger.warning(
                        "No model_name found in program metadata, "
                        "unable to update model selection algorithm."
                    )
                else:
                    parent = (
                        self.db.get(db_program.parent_id) if db_program.parent_id else None
                    )
                    baseline = parent.combined_score if parent else None
                    reward = db_program.combined_score if correct_val else None
                    model_name = db_program.metadata["model_name"]
                    result = self.llm_selection.update(
                        arm=model_name,
                        reward=reward,
                        baseline=baseline,
                    )
                    if result and self.verbose:
                        normalized_score, baseline = result

                        def fmt(x):
                            return f"{x:.4f}" if isinstance(x, (float, int)) else "None"

                        logger.debug(
                            f"==> UPDATED LLM SELECTION: model: "
                            f"{model_name.split('/')[-1][-25:]}..., "
                            f"score: {fmt(normalized_score)}, "
                            f"raw score: {fmt(reward)}, baseline: {fmt(baseline)}"
                        )
                        self.llm_selection.print_summary()

            self.db.save()
            self._update_best_solution()

        # Note: Meta summarization check is now done after completed generations
        # are updated in the main loop to ensure correct timing

        # Save meta memory state after each job completion
        self._save_meta_memory()

    def _update_best_solution(self):
        """Checks and updates the best program."""
        best_programs = self.db.get_top_programs(n=1, correct_only=True)
        if not best_programs:
            if self.verbose:
                logger.debug(
                    "No correct programs found yet, cannot determine best solution."
                )
            return

        best_program = best_programs[0]

        if best_program.id == self.best_program_id:
            return  # No change

        self.best_program_id = best_program.id

        source_dir = f"{self.results_dir}/{FOLDER_PREFIX}_{best_program.generation}"
        best_dir = Path(self.results_dir) / "best"

        if best_dir.exists():
            shutil.rmtree(best_dir)

        shutil.copytree(source_dir, best_dir)

        if self.verbose:
            logger.info(
                f"New best program found: gen {best_program.generation}, "
                f"id {best_program.id[:6]}... "
                f"Copied to {best_dir}"
            )

        # Record to unified scoring system immediately (crash protection)
        self._record_to_unified_scores(best_program, best_dir)

    def _record_to_unified_scores(self, best_program, best_dir: Path):
        """Record the best program to unified scoring system for crash protection.

        This ensures that even if evolution is interrupted, the best kernels
        found so far are recorded to scores/scores.csv and saved to scores/solutions/.
        """
        try:
            # Find project root by looking for scores/record_score.py
            results_path = Path(self.results_dir).resolve()
            project_root = results_path.parent

            # Check if this is a project with unified scoring
            record_script = project_root / "scores" / "record_score.py"
            if not record_script.exists():
                return  # Not a project with unified scoring

            # Get the kernel file
            kernel_file = best_dir / f"main.{self.lang_ext}"
            if not kernel_file.exists():
                return

            # Extract cycle count from program metrics
            cycles = None
            if best_program.public_metrics:
                import json
                try:
                    metrics = json.loads(best_program.public_metrics) if isinstance(best_program.public_metrics, str) else best_program.public_metrics
                    cycles = metrics.get('cycles')
                except:
                    pass

            if cycles is None:
                return  # Can't determine cycles

            # Call record_score.py with --only-if-best
            import subprocess
            result = subprocess.run(
                [sys.executable, str(record_script), '--only-if-best', str(kernel_file),
                 f"Evolution gen {best_program.generation}"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(project_root)
            )

            if 'RECORDED' in result.stdout:
                logger.info(f"NEW BEST recorded to unified scoring: {cycles} cycles")

        except Exception as e:
            # Don't let scoring failures affect evolution
            logger.debug(f"Could not record to unified scores: {e}")

    def run_patch(
        self,
        parent_program: Program,
        archive_programs: List[Program],
        top_k_programs: List[Program],
        generation: int,
        novelty_attempt: int = 1,
        resample_attempt: int = 1,
    ) -> tuple[Optional[str], dict, int]:
        """Run patch generation for a specific generation."""
        max_patch_attempts = self.evo_config.max_patch_attempts
        if self.verbose:
            logger.info(
                f"Edit Cycle {generation} -> {generation + 1}, "
                f"Max Patch Attempts: {max_patch_attempts}"
            )
        # Get current meta recommendations
        meta_recs, _, _ = self.meta_summarizer.get_current()
        # Construct edit / code change message
        patch_sys, patch_msg, patch_type = self.prompt_sampler.sample(
            parent=parent_program,
            archive_inspirations=archive_programs,
            top_k_inspirations=top_k_programs,
            meta_recommendations=meta_recs,
        )

        if patch_type in ["full", "cross"]:
            apply_patch = apply_full_patch
        elif patch_type == "diff":
            apply_patch = apply_diff_patch
        elif patch_type == "paper":
            raise NotImplementedError("Paper edit not implemented.")
            # apply_patch = apply_paper_patch
        else:
            raise ValueError(f"Invalid patch type: {patch_type}")

        total_costs = 0
        msg_history = []
        llm_kwargs = self.llm.get_kwargs()
        if self.llm_selection is not None:
            model_name = llm_kwargs["model_name"]
            self.llm_selection.update_submitted(model_name)
        code_diff = None  # Initialize code_diff
        num_applied_attempt = 0  # Initialize num_applied_attempt
        early_persist_metadata = {}  # Initialize early persistence metadata
        error_attempt = (
            "Max attempts reached without successful patch."  # Default error
        )
        patch_name = None
        patch_description = None
        output_path_attempt = None
        patch_txt_attempt = None
        patch_path = None
        diff_summary = {}

        for patch_attempt in range(max_patch_attempts):
            response = self.llm.query(
                msg=patch_msg,
                system_msg=patch_sys,
                msg_history=msg_history,
                llm_kwargs=llm_kwargs,
            )
            # print(response.content)
            if response is None or response.content is None:
                if self.verbose:
                    logger.info(
                        f"  PATCH ATTEMPT {patch_attempt + 1}/{max_patch_attempts} FAILURE. "
                        f"Error: LLM response content was None."
                    )
                # Prepare for next attempt or exit
                error_attempt = "LLM response content was None."
                num_applied_attempt = 0
                patch_txt_attempt = None
                if patch_attempt < max_patch_attempts - 1:
                    patch_msg = (
                        "The previous attempt to get an edit was not "
                        "successful because the LLM response was empty. "
                        "Try again."
                    )
                    if response:
                        msg_history = response.new_msg_history
                    continue
                else:  # Last attempt
                    break

            total_costs += response.cost  # Acc. cost

            # IMMEDIATE PERSISTENCE: Store LLM response metadata before evaluation
            # This ensures we don't lose responses if evaluation crashes
            early_persist_metadata = {
                "llm_response_archived": True,
                "llm_response_timestamp": datetime.now().isoformat(),
                "llm_model": response.model_name or llm_kwargs.get('model_name', 'unknown'),
                "llm_input_tokens": response.input_tokens,
                "llm_output_tokens": response.output_tokens,
                "llm_cost": response.cost,
                "llm_raw_content": response.content[:10000] if response.content else None,
                "generation": generation,
                "patch_attempt": patch_attempt + 1,
                "parent_id": parent_program.id if parent_program else None,
            }

            # IMMEDIATE FILE PERSISTENCE: Write to file before evaluation
            # This provides crash protection - the response is saved even if evaluation crashes
            if early_persist_metadata.get("llm_response_archived"):
                persist_dir = Path(self.results_dir) / f"{FOLDER_PREFIX}_{generation}"
                persist_dir.mkdir(parents=True, exist_ok=True)
                persist_path = persist_dir / f"llm_response_attempt_{patch_attempt + 1}.json"
                try:
                    with persist_path.open('w', encoding='utf-8') as f:
                        json.dump(early_persist_metadata, f, indent=2)
                    if self.verbose:
                        logger.debug(f"Persisted LLM response to {persist_path}")
                except Exception as e:
                    logger.warning(f"Failed to persist LLM response: {e}")

            patch_name = extract_between(
                response.content,
                "<NAME>",
                "</NAME>",
                False,
            )
            patch_description = extract_between(
                response.content,
                "<DESCRIPTION>",
                "</DESCRIPTION>",
                False,
            )

            # Apply the code patch (diff/full rewrite)
            (
                _,
                num_applied_attempt,
                output_path_attempt,
                error_attempt,
                patch_txt_attempt,
                patch_path,
            ) = apply_patch(
                original_str=parent_program.code,
                patch_str=response.content,
                patch_dir=f"{self.results_dir}/{FOLDER_PREFIX}_{generation}",
                language=self.evo_config.language,
                verbose=False,
            )

            if error_attempt is None and num_applied_attempt > 0:
                if patch_path:  # Ensure patch_path is not None
                    diff_summary = summarize_diff(
                        str(patch_path)
                    )  # Convert Path to str
                if self.verbose:
                    logger.info(
                        f"  PATCH ATTEMPT {patch_attempt + 1}/{max_patch_attempts} SUCCESS. "
                        f"Output: {output_path_attempt}, "
                        f"Patches Applied: {num_applied_attempt}."
                    )

                code_diff = patch_txt_attempt
                break  # Break from patch attempts
            else:
                error_str = (
                    str(error_attempt) if error_attempt else "No changes applied."
                )
                patch_msg = (
                    "The previous edit was not successful."
                    + " This was the error message: \n\n"
                    + error_str
                    + "\n\n Try again."
                )
                if self.verbose:
                    logger.info(
                        f"  PATCH ATTEMPT {patch_attempt + 1}/{max_patch_attempts} FAILURE. "
                        f"Error: '{error_str}', "
                        f"Patches Applied: {num_applied_attempt}."
                    )
                msg_history = response.new_msg_history
                code_diff = None
                if patch_attempt == max_patch_attempts - 1:  # Last attempt failed
                    # error_attempt is already set from apply_patch or default
                    pass

        # Only consider the diff summary for the original source file
        original_filename = f"original.{self.lang_ext}"
        if original_filename in diff_summary:
            diff_summary = diff_summary[original_filename]

        meta_edit_data = {
            # Early persist metadata (LLM response captured immediately after receipt)
            **early_persist_metadata,
            "patch_type": patch_type,
            "api_costs": total_costs,
            "num_applied": num_applied_attempt,
            "patch_name": patch_name,
            "patch_description": patch_description,
            "error_attempt": error_attempt,
            "novelty_attempt": novelty_attempt,
            "resample_attempt": resample_attempt,
            "patch_attempt": patch_attempt + 1,
            **llm_kwargs,
            "llm_result": response.to_dict() if response else None,
            "diff_summary": diff_summary,
        }
        if self.verbose and num_applied_attempt > 0:
            self._print_metadata_table(meta_edit_data, generation)
        # Delete generation from meta_edit_data
        return code_diff, meta_edit_data, num_applied_attempt

    def get_code_embedding(self, exec_fname: str) -> tuple[List[float], float]:
        """Get the embedding of the code."""
        # Read the evaluated code
        try:
            evaluated_code = Path(exec_fname).read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not read code for job {exec_fname}. Error: {e}")
            evaluated_code = ""
        if evaluated_code != "":
            # Get the embedding of the initial program
            try:
                if self.embedding is not None:
                    redacted_code = redact_immutable(evaluated_code, no_state=True)
                    if self.verbose:
                        logger.debug(
                            "=> EMBED: Code length - "
                            f"Original: {len(evaluated_code)} - "
                            f"Redacted: {len(redacted_code)}"
                        )

                    embedding_result, e_cost = self.embedding.get_embedding(
                        redacted_code
                    )
                else:
                    if self.verbose:
                        logger.debug("=> EMBED: No embedding model configured.")
                    embedding_result = []
                    e_cost = 0.0
                code_embedding = cast(List[float], embedding_result)
            except Exception as e:
                logger.warning(f"Could not embed code for job {exec_fname}. Error: {e}")
                code_embedding = []
                e_cost = 0.0
        else:
            code_embedding = []
            e_cost = 0.0
        return code_embedding, e_cost

    def _print_metadata_table(self, meta_data: dict, generation: int):
        """Display metadata in a formatted rich table."""
        # Create title with generation and attempt information
        title_parts = ["[bold magenta]Patch Metadata"]

        # Add generation if present
        if generation is not None:
            title_parts.append(
                f" - Gen {generation}/{self.evo_config.num_generations} - Novelty: {meta_data['novelty_attempt']}/{self.evo_config.max_novelty_attempts} - Resample: {meta_data['resample_attempt']}/{self.evo_config.max_patch_resamples} - Patch: {meta_data['patch_attempt']}/{self.evo_config.max_patch_attempts}"
            )

        # Add attempt information if present
        if all(
            key in meta_data
            for key in [
                "novelty_attempt",
                "resample_attempt",
                "patch_attempt",
                "generation",
            ]
        ):
            title_parts.append(
                f" (Novelty: {meta_data['novelty_attempt']}, "
                f"Resample: {meta_data['resample_attempt']}, "
                f"Patch: {meta_data['patch_attempt']})"
            )

        title_parts.append("[/bold magenta]")
        table = Table(
            title="".join(title_parts),
            show_header=True,
            header_style="bold cyan",
            border_style="magenta",
            box=rich.box.ROUNDED,
            width=120,  # Match display.py table width
        )
        table.add_column("Field", style="cyan bold", no_wrap=True, width=25)
        table.add_column("Value", style="green", overflow="fold", width=90)

        # Define display order and formatting for specific fields
        display_order = [
            "patch_type",
            "patch_name",
            "patch_description",
            "num_applied",
            "api_costs",
            "error_attempt",
        ]

        # Add ordered fields first
        for field_name in display_order:
            if field_name in meta_data:
                value = meta_data[field_name]
                if value is None:
                    formatted_value = "[dim]None[/dim]"
                elif field_name == "api_costs":
                    formatted_value = f"${value:.4f}"
                elif field_name == "error_attempt" and value is None:
                    formatted_value = "[green]Success[/green]"
                elif field_name == "error_attempt":
                    formatted_value = (
                        f"[red]{str(value)[:100]}...[/red]"
                        if len(str(value)) > 100
                        else f"[red]{value}[/red]"
                    )
                else:
                    formatted_value = str(value)

                table.add_row(field_name, formatted_value)

        # Add remaining fields (excluding llm_result, diff_summary, and header info)
        skip_fields = set(
            display_order
            + [
                "llm_result",
                "diff_summary",
                "generation",
                "novelty_attempt",
                "resample_attempt",
                "patch_attempt",
            ]
        )
        for field_key, field_value in meta_data.items():
            if field_key not in skip_fields:
                if field_value is None:
                    formatted_value = "[dim]None[/dim]"
                else:
                    formatted_value = (
                        str(field_value)[:100] + "..."
                        if len(str(field_value)) > 100
                        else str(field_value)
                    )
                table.add_row(field_key, formatted_value)

        # Add diff summary if available
        if "diff_summary" in meta_data and meta_data["diff_summary"]:
            diff_summary = meta_data["diff_summary"]
            if isinstance(diff_summary, dict):
                summary_text = ""
                for k, v in diff_summary.items():
                    summary_text += f"{k}: {v}; "
                table.add_row("diff_summary", summary_text.strip())
            else:
                table.add_row("diff_summary", str(diff_summary)[:200])

        self.console.print(table)

    def _save_meta_memory(self) -> None:
        """Save the meta memory state and LLM selection state to disk."""
        meta_memory_path = Path(self.results_dir) / "meta_memory.json"
        self.meta_summarizer.save_meta_state(str(meta_memory_path))
        # Also save LLM selection state to keep in sync
        self._save_llm_selection_state()

    def _restore_meta_memory(self) -> None:
        """Restore the meta memory state from disk."""
        meta_memory_path = Path(self.results_dir) / "meta_memory.json"

        if self.verbose:
            logger.info(f"Attempting to restore meta memory from: {meta_memory_path}")

        success = self.meta_summarizer.load_meta_state(str(meta_memory_path))
        if success:
            logger.info("Successfully restored meta memory state")
        else:
            if meta_memory_path.exists():
                logger.warning(
                    f"Meta memory file exists but failed to load: {meta_memory_path}"
                )
            else:
                logger.info("No previous meta memory state found - starting fresh")

    def _save_llm_selection_state(self) -> None:
        """Save the LLM selection bandit state to disk."""
        if self.llm_selection is None:
            return
        if not hasattr(self.llm_selection, 'to_dict'):
            return

        state_path = Path(self.results_dir) / "llm_selection_state.json"
        try:
            state = self.llm_selection.to_dict()
            with state_path.open('w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
            logger.debug(f"Saved LLM selection state to {state_path}")
        except Exception as e:
            logger.warning(f"Failed to save LLM selection state: {e}")

    def _restore_llm_selection_state(self) -> None:
        """Restore the LLM selection bandit state from disk."""
        if self.llm_selection is None:
            return
        if not hasattr(self.llm_selection, 'from_dict'):
            return

        state_path = Path(self.results_dir) / "llm_selection_state.json"
        if not state_path.exists():
            logger.info("No previous LLM selection state found - starting fresh")
            return

        try:
            with state_path.open('r', encoding='utf-8') as f:
                state = json.load(f)
            if self.llm_selection.from_dict(state):
                logger.info("Successfully restored LLM selection state")
                self.llm_selection.print_summary()
            else:
                logger.warning("LLM selection state exists but failed to load - starting fresh")
        except Exception as e:
            logger.warning(f"Failed to restore LLM selection state: {e}")
