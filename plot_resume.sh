trainCollection=tgif_sample2
valCollection=tgif_sample
testCollection=iacc.3_sample
concate=reduced
overwrite=1
resume=/root/VisualSearch/tgif_sample2/vatax-tgif_modality_0.001/msrvtt10kval/dual_encoding_concate_reduced_dp_0.2_measure_cosine/vocab_word_vocab_5_word_dim_500_text_rnn_size_512_text_norm_True_kernel_sizes_2-3-4_num_512/visual_feature_pyresnext-101_rbps13k,flatten0_output,os_visual_rnn_size_1024_visual_norm_True_kernel_sizes_2-3-4-5_num_512/mapping_text_0-2048_img_0-2048/loss_func_mrl_margin_0.2_direction_all_max_violation_True_cost_style_sum/optimizer_adam_lr_0.0001_decay_0.99_grad_clip_2.0_val_metric_recall/runs_0/model_best.pth.tar

#plot=True


# Generate a vocabulary on the training set
# ./do_get_vocab.sh $trainCollection

# training
gpu=0
CUDA_VISIBLE_DEVICES=$gpu python trainer.py $trainCollection $valCollection $testCollection  --overwrite $overwrite \
                                            --max_violation --text_norm --visual_norm --concate $concate --resume $resume --plot
                                            
# evaluation (Notice that a script file do_test_${testCollection}.sh will be automatically generated when the training process ends.)
./do_test_dual_encoding_${testCollection}.sh
