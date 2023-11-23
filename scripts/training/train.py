import datetime
import os
import warnings
import hydra
import omegaconf
import pytorch_lightning as pl
from loguru import logger
import scripts.training.lightning_modules as lm
from utils import get_latest_model_checkpoint_path


@hydra.main(
    config_path=os.path.join(os.getcwd(), "configs"), config_name="training_experiment"
)
@logger.catch
def main(configs: omegaconf.DictConfig) -> None:
    logger.add("train.log")
    logger.info("🚀 Training process started.")

    logger.info("📚 Creating dataset module.")

    classes=[]

    # classes = configs["dataset_module"]["classes"]
    # print("type(classes):",type(classes))
    # print("classes[]0:", classes[0])
    print(" ")
    print("  ------ ")
    for c in  configs["dataset_module"]["classes"]:
        classes.append(c)
    classes.append("background")
    print("classes", classes)

    class_weights=configs["dataset_module"]["class_weights"]
    class_weights=dict(class_weights)


    # x=y
    # Training data and model modules
    dataset_module = lm.HumanHeadSegmentationDataModule(
        dataset_root=configs.dataset_module.dataset_root,
        nn_image_input_resolution=configs.dataset_module.nn_image_input_resolution,
        batch_size=configs.dataset_module.batch_size,
        num_workers=configs.dataset_module.num_workers,
        all_augmentations=configs.dataset_module.all_augmentations,
        size_augmentation_keys=configs.dataset_module.size_augmentation_keys,
        content_augmentation_keys=configs.dataset_module.content_augmentation_keys,
    )

    logger.info("🕸 Creating neural network module.")
    # if os.path.isdir(configs.nn_module.model_save_path):
    if configs.training.continue_from_last_checkpoint=="true":
        last_checkpoint=get_latest_model_checkpoint_path(configs.hydra.run.dir)
        if last_checkpoint is not None:
            #load model from ckpt file.
            pass

    nn_module = lm.HumanHeadSegmentationModelModule(
        lr=configs.nn_module.lr,
        encoder_name=configs.nn_module.encoder_name,
        encoder_depth=configs.nn_module.encoder_depth,
        pretrained=configs.nn_module.use_pretrained,
        nn_image_input_resolution=configs.dataset_module.nn_image_input_resolution,
        classes=classes,
        class_weights=class_weights,
        load_last=None

    )

    # Callbacks
    logger.info("📲 Initializing callbacks.")
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor=configs.training.early_stop.monitor,
        patience=configs.training.early_stop.patience,
        mode=configs.training.early_stop.mode,
    )

    model_ckpt_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
        monitor=configs.training.early_stop.monitor,
        mode=configs.training.early_stop.mode,
        # fmt: off
        filename=configs.training.wandb_name + "-{epoch}-{" + configs.training.early_stop.monitor + ":.4f}",
        # fmt: on
        save_top_k=3,
        dirpath="./models",
        save_last=True,
    )

    # W&B Logger
    logger.info("📝 Initializing W&B logger.")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    wandb_logger = pl.loggers.WandbLogger(
        project=configs.training.wandb_project,
        name=f"{configs.training.wandb_name}-{timestamp}",
    )

    # Training env configs
    logger.info("🌍 Initializing training environment.")



    nn_trainer = pl.Trainer(
        logger=wandb_logger,
        log_every_n_steps=1,
        callbacks=[early_stop_callback, model_ckpt_callback],
        max_epochs=configs.training.max_epochs,
        gpus=1 if configs.training.with_gpu else 0,
    )

    # Train loop
    logger.info("🏋️ Starting training loop.")
    nn_trainer.fit(nn_module, dataset_module)

    # Display best model based on monitored metric
    logger.info(f"🥇 Best model: {model_ckpt_callback.best_model_path}")
    nn_module.load_from_checkpoint(model_ckpt_callback.best_model_path)

    # Test loop
    logger.info("🧪 Starting testing loop.")
    nn_trainer.test(nn_module, dataset_module)
    # current_working_directory = os.getcwd()
    # os.path.dirname(cwd)
    logger.success("🏁 Training process finished.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    main()
