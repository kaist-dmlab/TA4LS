## The cuurent hyper-parameters values are not necessarily the best ones for a specific risk.
def get_hparams_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

class HAR():
    def __init__(self):
        super(HAR, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5,

        }
        self.alg_hparams = {
            "SASA": {
                "domain_loss_wt": 7.3937939938562,
                "learning_rate": 0.005,
                "src_cls_loss_wt": 4.185814373345016,
                "weight_decay": 0.0001,
            },
            "CoDATS": {
                "domain_loss_wt": 3.2750474868706925,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 6.335109786953256,
                "weight_decay": 0.0001
            },
            "DANN": {
                "domain_loss_wt": 2.943729820531079,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 5.1390077646202,
                "weight_decay": 0.0001
            },
            "DIRT": {
                "cond_ent_wt": 1.20721518968644,
                "domain_loss_wt": 1.9012145515129044,
                "learning_rate": 0.005,
                "src_cls_loss_wt": 9.67861021290254,
                "vat_loss_wt": 7.7102843136045855,
                "weight_decay": 0.0001
            },
            "DSAN": {
                "learning_rate": 0.001,
                "mmd_wt": 2.0872340713147786,
                "src_cls_loss_wt": 1.8744909939900247,
                "domain_loss_wt": 1.59,
                "weight_decay": 0.0001
            },
            "HoMM": {
                "hommd_wt": 2.8305712579412683,
                "learning_rate": 0.0005,
                "src_cls_loss_wt": 0.1282520874653523,
                "domain_loss_wt": 9.13,
                "weight_decay": 0.0001
            },
        }


class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 128, # 128 ->64
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5,

        }
        self.alg_hparams = {
            "SASA": {
                "domain_loss_wt": 5.8045319155819515,
                "learning_rate": 0.005,
                "src_cls_loss_wt": 4.438490884851632,
                "weight_decay": 0.0001
            },
            "CoDATS": {
                "domain_loss_wt": 0.3551260369189456,
                "learning_rate": 0.005,
                "src_cls_loss_wt": 1.2534327517723889,
                "weight_decay": 0.0001
            },
            "DANN": {
                "domain_loss_wt": 0.27634197975549135,
                "learning_rate": 0.0005,
                "src_cls_loss_wt": 8.441929209893459,
                "weight_decay": 0.0001
            },
            "DIRT": {
                "cond_ent_wt": 1.7021814402136783,
                "domain_loss_wt": 1.6488583075821344,
                "learning_rate": 0.01,
                "src_cls_loss_wt": 6.427127521674593,
                "vat_loss_wt": 5.078600240648073,
                "weight_decay": 0.0001
            },
            "DSAN": {
                "learning_rate": 0.001,
                "mmd_wt": 5.01196798268099,
                "src_cls_loss_wt": 7.774381653453339,
                "domain_loss_wt": 6.708,
                "weight_decay": 0.0001
            },
            "HoMM": {
                "hommd_wt": 3.843851397373747,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 1.8311375304849091,
                "domain_loss_wt": 1.102,
                "weight_decay": 0.0001
            },
        }


class WISDM():
    def __init__(self):
        super().__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5,

        }
        self.alg_hparams = {
            "SASA": {
                "domain_loss_wt": 1.2632988839197083,
                "learning_rate": 0.005,
                "src_cls_loss_wt": 9.898676755625807,
                "weight_decay": 0.0001
            },
            "HoMM": {
                "hommd_wt": 6.799448304230478,
                "learning_rate": 0.005,
                "src_cls_loss_wt": 0.2563533185103576,
                "domain_loss_wt": 4.239,
                "weight_decay": 0.0001
            },
            "DANN": {
                "domain_loss_wt": 2.6051391453662873,
                "learning_rate": 0.005,
                "src_cls_loss_wt": 5.272383517138417,
                "weight_decay": 0.0001
            },
            "DIRT": {
                "cond_ent_wt": 1.6935884891647972,
                "domain_loss_wt": 7.774841143071709,
                "learning_rate": 0.005,
                "src_cls_loss_wt": 9.62463958771893,
                "vat_loss_wt": 4.644539486962429,
                "weight_decay": 0.0001
            },
            "CoDATS": {
                "domain_loss_wt": 4.574872968982744,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 5.860885469514424,
                "weight_decay": 0.0001
            },
            "DSAN": {
                "learning_rate": 0.005,
                "mmd_wt": 1.5468030830413808,
                "src_cls_loss_wt": 1.2981011362021273,
                "domain_loss_wt": 0.1,
                "weight_decay": 0.0001
            },
        }


class HHAR_SA():
    def __init__(self):
        super().__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5,
        }
        self.alg_hparams = {
            "SASA": {
                "domain_loss_wt": 5.760124609738364,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 4.130742585941761,
                "weight_decay": 0.0001
            },
            "DSAN": {
                "learning_rate": 0.0005,
                "mmd_wt": 0.5993593617252002,
                "src_cls_loss_wt": 0.386167577207679,
                "domain_loss_wt": 0.16,
                "weight_decay": 0.0001
            },
            "CoDATS": {
                "domain_loss_wt": 9.314114040099962,
                "learning_rate": 0.0005,
                "src_cls_loss_wt": 7.700018679383289,
                "weight_decay": 0.0001
            },
            "HoMM": {
                "hommd_wt": 7.172430927893522,
                "learning_rate": 0.0005,
                "src_cls_loss_wt": 0.20121211752349172,
                "domain_loss_wt": 0.9824,
                "weight_decay": 0.0001
            },
            "DIRT": {
                "cond_ent_wt": 1.329734510542011,
                "domain_loss_wt": 6.632293308809388,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 7.729881324550688,
                "vat_loss_wt": 6.912258476982827,
                "weight_decay": 0.0001
            },
            "DANN": {
                "domain_loss_wt": 1.0296390274908802,
                "learning_rate": 0.0005,
                "src_cls_loss_wt": 2.038458138479581,
                "weight_decay": 0.0001
            },

        }
