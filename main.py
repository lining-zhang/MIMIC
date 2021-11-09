from trainer import Transformer_Encoder_Model
from data_dealer import yaml_config


config = yaml_config()


model = Transformer_Encoder_Model(dim=8,
                                  max_len=95,
                                  vocab_size=5913,
                                  pad_index=2,
                                  num_layers=2,
                                  ffn_hidden=64,
                                  n_head=8,
                                  drop_prob=0.1,
                                  eps=1e-6)


def main():
    nmt = NMT(config)
    transformer_model = Transformer(dim=config["parameters"]["d_model"],
                                    nhead=config["parameters"]["n_head"],
                                    num_encoder_layers=config["parameters"]["enc_layers"],
                                    num_decoder_layers=config["parameters"]["dec_layers"],
                                    dim_feedforward=config["parameters"]["hidden_size"],
                                    dropout=config["parameters"]["dropout_rate"],
                                    src_vocab_size=nmt.source_vocab,
                                    tgt_vocab_size=nmt.target_vocab)
    model = nmt.load_model(transformer_model)
    nmt.train(model, if_eval=True)

if __name__ == "__main__":
    main()