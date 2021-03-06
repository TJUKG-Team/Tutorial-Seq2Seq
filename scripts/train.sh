cd `dirname $0`; cd ..;
echo -e "The current working directory is $(pwd)\n"

python Seq2Seq_train.py --model_name LSTM --dataset Multi30k --train_from_zero
# python Seq2Seq_train.py --model_name GRU --dataset Multi30k
# python Seq2Seq_train.py --model_name GRU_Attention --dataset Multi30k

# python Seq2Seq_train.py --model_name LSTM --dataset DailyDialog --batch_size 16 --train_from_zero
# python Seq2Seq_train.py --model_name GRU --dataset DailyDialog --batch_size 16
# python Seq2Seq_train.py --model_name GRU_Attention --dataset DailyDialog --batch_size 4
