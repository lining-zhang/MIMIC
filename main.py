from data_dealer import yaml_config
from trainer import Trainer, Transformer_Encoder_Model
from precision_calculation import load_attn_pickle, load_vocab_notes, load_label, \
                                  all_record_result, all_record_counter, calculate_precision


config = yaml_config()


def main():
    trainer = Trainer(config)
    transformer_encoder_model = Transformer_Encoder_Model(dim=config["embed_dim"],
                                                          max_len=trainer.max_len,
                                                          vocab_size=len(trainer.dictionary_obj),
                                                          pad_index=config["pad_index"],
                                                          num_layers=config["num_layers"],
                                                          ffn_hidden=config["ffn_hidden"],
                                                          n_head=config["n_head"],
                                                          drop_prob=config["drop_prob"],
                                                          eps=config["eps"])
    model = trainer.load_model(transformer_encoder_model)
    trainer.train(model)

    trainer.eval(model, "test")
    attention = load_attn_pickle(config["attn_path"] + "/attentions.pkl")
    vocab_notes = load_vocab_notes(config["test_path"], config["vocab_obj_path"])
    label_dict = load_label(file_path=config["label_path"], upper=55000)

    result_dict = all_record_result(attention, vocab_notes, config["pair_json_path"])
    counter_dict, len_list = all_record_counter(result_dict, label_dict)
    precision_dict = calculate_precision(counter_dict, len_list)

    for key, value in precision_dict.items():
        print("===== Precision =====")
        print("For {}, the precision is {}".format(key, value))


if __name__ == "__main__":
    main()