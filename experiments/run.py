import argparse
from experiments.experiment import Experiment

sessions = ["Mouse12-120806"]
    # , "Mouse12-120807", "Mouse12-120809", "Mouse12-120810", "Mouse17-130125",
    #         "Mouse17-130128", "Mouse17-130129", "Mouse17-130130", "Mouse17-130131", "Mouse17-130201",
    #         "Mouse17-130202", "Mouse17-130203", "Mouse17-130204", "Mouse20-130514", "Mouse20-130515",
    #         "Mouse20-130516", "Mouse20-130517", "Mouse24-131216", "Mouse24-131217", "Mouse24-131218",
    #         "Mouse25-140130", "Mouse25-140131", "Mouse25-140204", "Mouse28-140311", "Mouse28-140313",
    #         "Mouse28-140317", "Mouse28-140318"]

for session in sessions:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data/", type=str,
                        help="root directory all the data will be stored in")
    parser.add_argument("--save_dir", default="../saved_models/", type=str,
                        help="root directory model checkpoints will be saved in")
    parser.add_argument("--res_dir", default="../results/graphs/", type=str,
                        help="root directory results will be saved in")
    parser.add_argument("--dataset", default="spike", help="mnist | fmnist | kmnist")
    parser.add_argument("--enc_layers", nargs="+", type=int, help="encoder layers", default="100")
    parser.add_argument("--dec_layers", nargs="+", type=int, help="decoder layers", default="100")
    parser.add_argument("--model", default="RVAE", help="RVAE | VAE")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size")
    parser.add_argument("--warmup_learning_rate", default=1e-5, type=float, help="p_mu learning rate")
    parser.add_argument("--sigma_learning_rate", default=1e-5, type=float, help="p_sigma learning rate")
    parser.add_argument("--mu_epochs", default=40, type=int, help="number of training epochs (decoder mu)")
    parser.add_argument("--sigma_epochs", default=40, type=int, help="number of training epochs (decoder sigma)")
    parser.add_argument("--device", default="cuda:3", type=str, help="cuda | cpu")
    parser.add_argument("--seed", default=None, help="random seed")
    parser.add_argument("--log_invl", default=100, type=int,
                        help="the interval in which training stats will be reported")
    parser.add_argument("--save_invl", default=25, type=int, help="the interval in which model weights will be saved")
    parser.add_argument("--latent_dim", default=3, type=int, help="dimensionality of latent space")
    parser.add_argument("--num_centers", default=64, type=int,
                        help="number of centers for the RBF regularization in the decoder sigma net")
    parser.add_argument("--rbf_beta", default=0.01, type=float, help="rbf layer beta parameter")
    parser.add_argument("--rec_b", default=1e-9, type=float)
    parser.add_argument("--num_components", default=128, type=int, help="number of components for the prior")
    parser.add_argument("--ckpt_path", default=None, type=str)
    parser.add_argument("--session", default=session, type=str, help="experimental session id")
    # parser.add_argument("--ckpt_path", default="../saved_models/RVAE/spike_epoch40_"+session+"_sub.ckpt", type=str)
    # parser.add_argument("--ckpt_path", default="../saved_models/VAE/spike_K1epoch40_"+session+".ckpt", type=str)
    args = parser.parse_args()

    experiment = Experiment(args)
    print("Session: " + session + "******************************************************")
    if args.ckpt_path is None:
        experiment.train()
        experiment.eval()
    else:
        experiment.eval(args.ckpt_path)
        experiment.visualize(args.ckpt_path, args.res_dir, 3)

    break
