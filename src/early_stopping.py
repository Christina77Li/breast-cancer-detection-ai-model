class EarlyStopping:
    def __init__(self, patience=10, min_delta=1E-4, start_epoch=10, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.start_epoch = start_epoch
        self.mode = mode.lower()
        self.best_score = None
        self.num_bad_epochs = 0
        self.early_stop = False
        self.save_best = False

    def __call__(self, metric, epoch):
        if epoch < self.start_epoch:
            return
        if self.best_score is None:
            self.best_score = metric
            self.save_best = True
            return

        # min or max
        improvement = (
            metric - self.best_score if self.mode == "max" else self.best_score - metric
        )

        if improvement > self.min_delta:
            self.best_score = metric
            self.num_bad_epochs = 0
            self.save_best = True
        else:
            self.num_bad_epochs += 1
            self.save_best = False

        if self.num_bad_epochs >= self.patience:
            self.early_stop = True
