OUTPATH=/raid/yihui/review_analysis
PREPROCESS_CODE_PATH=/home/yihui/flair/shopee_review_preprocess.py
TRAIN_CODE_PATH=/home/yihui/flair/shopee_review_train.py
CODE_FILE=/home/yihui/flair
##### remove old training corpus because new data will be appended at the end of the file
for category in $(seq 1 5); do
  for split in train dev test; do
    FILE=$OUTPATH/"$category"_$split.txt
    if [ -d "$FILE" ]; then
        rm -r $FILE
        rm $OUTPATH/"$category"_$split.txt
    else
        echo "$FILE does not exist."
    fi
  done
done

##### data preprocessing and prepare flair training data that meets flair's format
python $PREPROCESS_CODE_PATH

##### concatenating training data (pyspark saves dataframe into a directory)
##### shuffling data to mix all types of data together
for category in $(seq 1 5); do
  for split in train dev test; do
    FILE=$OUTPATH/"$category"_$split
    if [ -d "$FILE" ]; then
        echo "$FILE shuffling..."
        cat $FILE/* | shuf >> $FILE.txt
        echo "$FILE "
        cat $FILE.txt | wc -l
        rm -r $FILE
    else
        echo "$FILE does not exist."
    fi
  done
done

##### remove pyspark-producing directories
for category in $(seq 1 5); do
    FILE=$CODE_FILE/$category.out
    if [ -f "$FILE" ]; then
        rm $FILE
    else
        echo "$FILE does not exist."
    fi
done


nohup python $TRAIN_CODE_PATH --category 0 --which_gpu 0 &> $CODE_FILE/1.out&
nohup python $TRAIN_CODE_PATH --category 1 --which_gpu 1 &> $CODE_FILE/2.out&
nohup python $TRAIN_CODE_PATH --category 2 --which_gpu 2 &> $CODE_FILE/3.out&
nohup python $TRAIN_CODE_PATH --category 3 --which_gpu 5 &> $CODE_FILE/4.out&
nohup python $TRAIN_CODE_PATH --category 4 --which_gpu 6 &> $CODE_FILE/5.out&
```
for no_gpu in $(seq 1 5); do
    let "category_id=$no_gpu-1"
    nohup python $TRAIN_CODE_PATH --which_gpu $no_gpu --category $category_id &> $CODE_FILE/$no_gpu.out&
done
```
