cd `dirname $0`; cd ..;
echo -e "The current working directory is $(pwd)\n"

tensorboard --logdir=logs