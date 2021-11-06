#!/bin/bash
#rm -r mlaas
#git clone https://git.tapsell.ir/brain/mlaas.git
#git pull
#rm -r tempenv
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
/root/software/anaconda3/bin/virtualenv tempenv
source tempenv/bin/activate
pip install $DIR/../../
python $DIR/read_cafe_video_data.py && now=$(date +%Y-%m-%dT%H:%M:%S) && curl --location --request GET "http://162.245.81.148:31010/telegram/bot567886937:AAER13EmjCu4K20O_IG5FrdtjEzAnxUpMQ0/sendMessage?chat_id=-452873691&text='server-time:"$now"'%D8%AF%D8%A7%D8%AF%D9%87%E2%80%8C%D9%87%D8%A7%20%D8%A8%D8%A7%20%D9%85%D9%88%D9%81%D9%82%DB%8C%D8%AA%20%D8%AE%D9%88%D8%A7%D9%86%D8%AF%D9%87%20%D8%B4%D8%AF%D9%86%D8%AF%20%E2%9C%85" \
--header 'Authorization: JhbGciOiJIUzsiYW5hbHl0aWNzX3NhdXJvbiJdLCJzY29wZSI6WyJyZWFkIiwiXJ2'
python $DIR/learn_cafe_video_model.py && now=$(date +%Y-%m-%dT%H:%M:%S) && curl --location --request GET "http://162.245.81.148:31010/telegram/bot567886937:AAER13EmjCu4K20O_IG5FrdtjEzAnxUpMQ0/sendMessage?chat_id=-452873691&text='server-time:"$now"'%0A%D8%A8%D8%A7%20%D9%85%D9%88%D9%81%D9%82%DB%8C%D8%AA%20%D8%A7%D8%AC%D8%B1%D8%A7%20%D8%B4%D8%AF%20%D8%A7%DB%8C%D9%86%20%D8%A8%D8%A7%D8%B1%20%D9%87%D9%85%20%E2%9C%85%0A%0A" \
--header 'Authorization: JhbGciOiJIUzsiYW5hbHl0aWNzX3NhdXJvbiJdLCJzY29wZSI6WyJyZWFkIiwiXJ2'
