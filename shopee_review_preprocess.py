from pyspark import *
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from nltk.tokenize import TweetTokenizer, word_tokenize
import emoji
import re
# import nlpaug.augmenter.char as nac
# import nlpaug.augmenter.word as naw

try:
    ######## defining pyspark
    sc = SparkContext("local", "pami")
    sqlContext = SparkSession.builder.getOrCreate()
    print('created sqlContext')
except:
    pass

attributes = [{'name':'review_id','datatype':'numerical'},\
        {'name':'review','datatype':'string'},\
        {'name':'rating','datatype':'numerical'}]

schema = StructType([StructField(att['name'], StringType()) if att['datatype']=='string' else StructField(att['name'], IntegerType()) for att in attributes])

def emoji_cleaning(text=None):
    if text is not None or text.strip():
        # Change emoji to text
        text = emoji.demojize(text).lower().replace(":", " ")
        # Delete repeated emoji
        words = word_tokenize(text)
        repeated_list = []
        for word in words:
            if word not in repeated_list:
                repeated_list.append(word)
        
        text = ' '.join(text for text in repeated_list)
        text = text.replace("_", " ").replace("-", " ")
        return text

def review_cleaning(text=None):
    if text is not None or text.strip():
        # delete lowercase and newline
        text = text.lower()
        text = re.sub(r'\n', '', text)
        # change emoticon to text :) :( xd lol
        text = re.sub(r': *\(+\b', 'dislike', text)
        text = re.sub(r'\(*[￣‵]+﹏+[￣′]+\)*', 'dislike', text)
        text = re.sub(r'[:;=]+ *\)+\b', 'smile', text)
        text = re.sub(r'\b[Xx]+ *[Dd]+', 'smile', text)
        text = re.sub(r'\b[Ll]+ *[Oo]+ *[Ll]+\b', 'smile', text)
        text = re.sub(r'\(*[‵￣]+[︶ˇ▽]+[′￣]+\)*','smile',text)
        # delete punctuation
        text = re.sub('[^a-z0-9 ]', ' ', text)
        # eliminate mulitple spaces
        text = re.sub(' +', ' ',text)
        return text

def delete_repeated_char(text=None):
    if text is not None:
        if text.strip():
            # delete characters repeated more than twice
            text = re.sub(r'(\w)\1{2,}', r'\1', text)
            if text:
                return text

def write_text_file(data, entity_type):
    N = len(data)
    indices = [('train',0,int(N*0.9)), ('dev',int(N*0.9),int(N*0.95)), ('test',int(N*0.95),N)]
    review_score = int(entity_type[-1])
    for file_cat, head, tail in indices:
        filename = "/raid/yihui/review_analysis/%s_%s.txt"%(entity_type,file_cat)
        fd = open(filename,'w')
        print('saving %s data to %s'%(file_cat,filename))
        for label, text in data[head:tail]:
            if label == review_score:
                print('__label__%s %s'%(entity_type,text),file = fd)
            else:
                print('__label__%s %s'%('NONE',text),file = fd)
        fd.close()
    return

def concat_label_sentence(text,rating,target):
    if rating==target:
        return "__label__%d %s"%(target, text)
    else:
        return "__label__NONE %s"%(text)

if __name__ == '__main__':
    input_json = '/raid/yihui/review_analysis/train.csv'
    print('reading file from %s'%(input_json))
    input_df = sqlContext.read.schema(schema).option("mode", "DROPMALFORMED").csv(input_json)
    ######## storing training data for each sentiment
    udf_tokenizer = udf(lambda text: delete_repeated_char(review_cleaning(emoji_cleaning(text))), StringType())
    input_df = input_df.withColumn('text', udf_tokenizer('review'))
    input_df = input_df.where(length(input_df.text)>0)
    # split train, dev, test ratio
    indices = [('train',0,0.9), ('dev',0.9,0.95), ('test',0.95,1)]
    # insert_aug = naw.ContextualWordEmbsAug(model_path='roberta-base', action='insert')
    train_range={1:(1,2),2:(1,3),3:(2,4),4:(3,5),5:(4,5)}
    for target_rating in range(1,6):
        column = 'rating%d'%(target_rating)
        udf_training_sentence = udf(lambda text, rt: concat_label_sentence(text,rt,target_rating), StringType())
        training_sentence_df = input_df.filter(input_df.rating.between(train_range[target_rating][0], train_range[target_rating][1]))\
                                .withColumn(column, udf_training_sentence('text','rating'))\
                                .select(column)
        target_df = training_sentence_df.where(input_df.rating==target_rating)
        rest_df = training_sentence_df.where(input_df.rating!=target_rating)
        N_target = target_df.count()
        N_rest = rest_df.count()
        # udf_augmentation = udf(lambda text: insert_aug.augment(text), StringType())
        # all_aug_list=[]
        # if N_rest//N_target > 2:
        #     times = (N_rest//N_target)-1
        #     all_aug_df=target_df
        #     original_sentence_df = input_df.withColumn(column, udf_training_sentence('text','rating')).select('text').where(input_df.rating==target_rating)
        #     for _ in range(times):
        #         aug_df = original_sentence_df.withColumn(column, udf_augmentation('text')).select(column)
        #         all_aug_df = all_aug_df.union(aug_df)
        #     if all_aug_df is not None:
        #         target_df = all_aug_df.withColumn(column, udf_tokenizer(column)).dropDuplicate()
        targets = target_df.collect()
        rests = rest_df.collect()
        # N_target = target_df.count()
        print('starting to write %s training data'%(column))
        for file_cat, head, tail in indices:
            filename = "/raid/yihui/review_analysis/%d_%s"%(target_rating,file_cat)
            sc.parallelize(targets[int(N_target*head):int(N_target*tail)]).toDF().coalesce(1).write.format("text").option("header", "false").mode("append").save(filename)
            sc.parallelize(rests[int(N_rest*head):int(N_rest*tail)]).toDF().coalesce(1).write.format("text").option("header", "false").mode("append").save(filename)
            print('saving to %s'%(filename))
