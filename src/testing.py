from transformers import PreTrainedModel

from datasets import DatasetDict

from src.log_writer import LogWriter
from src.generation import Generator
from src.training_config import TrainingConfig
from src.validation_metrics import (
    ParsabilityRate,
    CoverageTest,
    GeomValidityRate,
    RoomsOverlappingTest,
    RequiredRoomsTest
)


def test(
        model: PreTrainedModel,
        tokenizer, dataset: DatasetDict,
        config: TrainingConfig,
        log_writer: LogWriter
    ):
    pars_rate = ParsabilityRate()
    validity_rate = GeomValidityRate()
    cov_rate = CoverageTest()
    room_overlap_rate = RoomsOverlappingTest()
    required_rooms = RequiredRoomsTest()

    generator = Generator(model, tokenizer, dataset)

    for batch in generator.generate_in_batches():
        floor_plans = pars_rate.parse(batch)
        floor_plans = validity_rate.filter_out_invalid(floor_plans)
        cov_rate.measure(floor_plans)
        room_overlap_rate.measure(floor_plans)
        required_rooms.measure(floor_plans)
    
    hparams = config.train_config.copy()
    hparams = hparams | config.model_config.copy()

    hparams.pop("eval_steps", None)
    hparams.pop("log_comment", None)
    hparams.pop("name", None)
    hparams.pop("lr_scheduler", None)

    metrics = {}

    pars_rate.add_to_metrics(metrics)
    validity_rate.add_to_metrics(metrics)
    cov_rate.add_to_metrics(metrics)
    room_overlap_rate.add_to_metrics(metrics)
    required_rooms.add_to_metrics(metrics)

    log_writer.add_hparams(hparams, metrics)
