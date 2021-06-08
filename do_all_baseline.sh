rootpath=/root/VisualSearch

trainCollection=vatex
targettrainCollection=msrvtt10ktrain
valCollection=msrvtt10kval
testCollection=msrvtt10ktest
concate=reduced
overwrite=1
cv_name=baseline
n_caption=20

# build vocab
./do_get_vocab.sh $trainCollection

# training
gpu=0
CUDA_VISIBLE_DEVICES=$gpu python trainer.py --trainCollection $trainCollection --targettrainCollection $targettrainCollection --valCollection $valCollection --testCollection $testCollection  --overwrite $overwrite --rootpath $rootpath \
                                            --max_violation --text_norm --visual_norm --concate $concate --n_caption $n_caption\
                                            --cv_name $cv_name 
                                            
# evaluation (Notice that a script file do_test_${testCollection}.sh will be automatically generated when the training process ends.)
./do_test_dual_encoding_${testCollection}.sh
