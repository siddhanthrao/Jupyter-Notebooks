from flask import Flask, render_template, url_for, flash, redirect,request
from forms import RegistrationForm, LoginForm
import os
import subprocess
import sys
import pandas
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import pickle
import sys
import pycats
import datetime
import os
import glob
import time
import csv
import re
import numpy as np
import pandas as pd
import pyhive
from pyhive import hive
from nltk.stem.wordnet import WordNetLemmatizer
from datetime import datetime
import matplotlib
matplotlib.use('agg')
import matplotlib as plt
#plt.switch_backend('count')
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import seaborn as sns
from nltk.stem.wordnet import WordNetLemmatizer
from pylab import *

#global dbname
#dbname = "abc"
#db = SQLAlchemy(app)
app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
#var1 = []

@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['GET', 'POST'])
def home():
    global dbname
    global filter
    form = RegistrationForm()
    if request.method == "POST":
        unm = request.form.get("username", None)
        un = str(unm)
        cwd = os.getcwd()
        filename = os.path.expanduser('.') + '/static/q18.jpg'
        print cwd
        print filename
        def text2int (textnum, numwords={}):
            if not numwords:
                units = [
                "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
                "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
                "sixteen", "seventeen", "eighteen", "nineteen",
                ]

                tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

                scales = ["hundred", "thousand", "million", "billion", "trillion"]

                numwords["and"] = (1, 0)
                for idx, word in enumerate(units):  numwords[word] = (1, idx)
                for idx, word in enumerate(tens):       numwords[word] = (1, idx * 10)
                for idx, word in enumerate(scales): numwords[word] = (10 ** (idx * 3 or 2), 0)

            ordinal_words = {'first':1, 'second':2, 'third':3, 'fifth':5, 'eighth':8, 'ninth':9, 'twelfth':12}
            ordinal_endings = [('ieth', 'y'), ('th', '')]

            textnum = textnum.replace('-', ' ')

            current = result = 0
            curstring = ""
            onnumber = False
            for word in textnum.split():
                if word in ordinal_words:
                    scale, increment = (1, ordinal_words[word])
                    current = current * scale + increment
                    if scale > 100:
                        result += current
                        current = 0
                    onnumber = True
                else:
                    for ending, replacement in ordinal_endings:
                        if word.endswith(ending):
                            word = "%s%s" % (word[:-len(ending)], replacement)

                    if word not in numwords:
                        if onnumber:
                            curstring += repr(result + current) + " "
                        curstring += word + " "
                        result = current = 0
                        onnumber = False
                    else:
                        scale, increment = numwords[word]

                        current = current * scale + increment
                        if scale > 100:
                            result += current
                            current = 0
                        onnumber = True

            if onnumber:
                curstring += repr(result + current)

            return curstring

        ### Creating a column name dictionary
        
        ### Exrtacting dates --- not run

        def date_extract(x):

            numeric_dates = re.findall(r'\d+\S\d+\S\d+', x) 
            string_dates = re.findall(r'[A-Z]\w+\s\d+', x)

            len_numeric_dates = len(numeric_dates)
            len_string_dates = len(string_dates)

            if len_numeric_dates == 0 and len_string_dates == 0 :
                dates = [0,'','']
            elif (len_numeric_dates + len_string_dates) > 2 :
                dates = [3,'','']
            elif (len_numeric_dates + len_string_dates) == 1 :
                if len_numeric_dates == 1 :
                    date1 = dp.parse(numeric_dates[0])
                if len_string_dates == 1 :
                    date1 = dp.parse(string_dates[0])
                dates = [1,date1,'']
            elif len_numeric_dates == 2 :
                date1 = dp.parse(numeric_dates[0])
                date2 = dp.parse(numeric_dates[1])
                if date1 > date2 :
                    dates = [2,date2,date1]
                elif date1 < date2 :
                    dates = [2,date1,date2]
                elif date1 == date2 :
                    dates = [1,date1,'']
            elif len_string_dates == 2 :
                date1 = dp.parse(string_dates[0])
                date2 = dp.parse(string_dates[1])
                if date1 > date2 :
                    dates = [2,date2,date1]
                elif date1 < date2 :
                    dates = [2,date1,date2]
                elif date1 == date2 :
                    dates = [1,date1,'']     
            elif (len_numeric_dates + len_string_dates) == 2 :
                date1 = dp.parse(numeric_dates[0])
                date2 = dp.parse(string_dates[0])
                if date1 > date2 :
                    dates = [2,date2,date1]
                elif date1 < date2 :
                    dates = [2,date1,date2]
                elif date1 == date2 :
                    dates = [1,date1,'']
            return dates

        ### Pre_post Extract--- not run

        def pre_post_extract(x):

            numeric_dates = re.findall(r'\d+\S\d+\S\d+', x) 
            string_dates = re.findall(r'[A-Z]\w+\s\d+', x)

            x = x.replace("  "," ")

            for i in numeric_dates:
                x = x.replace(i,"date_num")

            for i in string_dates:
                x = x.replace(i,"date_str")

            pre_post = []

            if(x.find("before date_") != -1):
                pre_post.append("before")

            if(x.find("after date_") != -1):
                pre_post.append("after")

            if(x.find("between date_") != -1):
                pre_post.append("between")

            if(x.find("post date_") != -1):
                pre_post.append("post")        

            return pre_post

        ### Date Replace

        def date_replace(x):

            numeric_dates = re.findall(r'\d+\S\d+\S\d+', x) 
            string_dates = re.findall(r'[A-Z]\w+\s\d+', x)

            x = x.replace("  "," ")

            for i in numeric_dates:
                x = x.replace(i,"date_num")

            for i in string_dates:
                x = x.replace(i,"date_str")

            return x

        ### Replacing essential n-grams to single words

        def replace_ess(x):

            x = x.replace('  ', ' ')

            ### Replacing ' ' with '_' needs to be automated for regular updated for scaling
            x = x.replace('detection method', 'detection_method')
            x = x.replace('adjd disposition clm lvl', 'adjd_disposition_clm_lvl')
            x = x.replace('adjd disposition claim level', 'adjd_disposition_clm_lvl')
            x = x.replace('adjd disposition claim lvl', 'adjd_disposition_clm_lvl')
            x = x.replace('claim number', 'claim_number')
            x = x.replace('claim type', 'claim_type')
            x = x.replace('fwae dipsosition', 'fwae_dipsosition')
            x = x.replace('procedure code', 'procedure_code')
            x = x.replace('uhn region', 'uhn_region')
            x = x.replace('uhnregion', 'uhn_region')
            x = x.replace('prov spcl catgy desc', 'prov_spcl_catgy_desc')
            x = x.replace('prov splt catgy desc', 'prov_spcl_catgy_desc')
            x = x.replace('provider speciality category description', 'prov_spcl_catgy_desc')
            x = x.replace('provider speciality', 'prov_spcl_catgy_desc')
            x = x.replace('speciality category', 'prov_spcl_catgy_desc')
            x = x.replace('provider speciality category', 'prov_spcl_catgy_desc')
            x = x.replace('speciality category description', 'prov_spcl_catgy_desc')
            x = x.replace("e&i nat'l accts.", "e&i_nat'l_accts.")
            x = x.replace('west region', 'west_region')
            x = x.replace('central region', 'central_region')
            x = x.replace('northeast region', 'northeast_region')
            x = x.replace('southeast region', 'southeast_region')
            x = x.replace('leased markets', 'leased_markets')
            x = x.replace('colon and rectal surgery', 'colon_and_rectal_surgery')
            x = x.replace('independent lab', 'independent_lab')
            x = x.replace('social worker', 'social_worker')
            x = x.replace('family practice/clinic', 'family_practice/clinic')
            x = x.replace('rehabilitation medicine', 'rehabilitation_medicine')
            x = x.replace('blood banking', 'blood_banking')
            x = x.replace('emergency medicine', 'emergency_medicine')
            x = x.replace('physical/occupational therapy', 'physical/occupational_therapy')
            x = x.replace('rn special service', 'rn_special_service')
            x = x.replace('medical supply firm', 'medical_supply_firm')
            x = x.replace('pulmonary disease', 'pulmonary_disease')
            x = x.replace('special provider agreement', 'special_provider_agreement')
            x = x.replace('home health', 'home_health')
            x = x.replace('family practice', 'family_practice')
            x = x.replace('plastic surgeon', 'plastic_surgeon')
            x = x.replace('podiatrist - non-md', 'podiatrist_-_non-md')
            x = x.replace('internal medicine specialist', 'internal_medicine_specialist')
            x = x.replace('speech therapist', 'speech_therapist')
            x = x.replace('vascular surgeon', 'vascular_surgeon')
            x = x.replace('after hours clinic/urgent care', 'after_hours_clinic/urgent_care')
            x = x.replace('thoracic surgeon', 'thoracic_surgeon')
            x = x.replace('infectious disease sepcialist', 'infectious_disease_sepcialist')
            x = x.replace('therapeutic radiology', 'therapeutic_radiology')
            x = x.replace('family practice specialist', 'family_practice_specialist')
            x = x.replace('accidental dental/medical dent', 'accidental_dental/medical_dent')
            x = x.replace('home health/home iv', 'home_health/home_iv')
            x = x.replace('pediatric specialist', 'pediatric_specialist')
            x = x.replace('out of area hosp nonpar', 'out_of_area_hosp_nonpar')
            x = x.replace('mh/cd outpatient facility', 'mh/cd_outpatient_facility')
            x = x.replace('nuclear medicine', 'nuclear_medicine')
            x = x.replace('ob gynecologist spec', 'ob_gynecologist_spec')
            x = x.replace('licensed practical nurse', 'licensed_practical_nurse')
            x = x.replace('skilled nursing facility', 'skilled_nursing_facility')
            x = x.replace('medical records not received', 'medical_records_not_received')
            x = x.replace('paid after investigation', 'paid_after_investigation')
            x = x.replace('denied after investigation', 'denied_after_investigation')
            x = x.replace('prompt paid', 'prompt_paid')
            x = x.replace('unknown', 'unknown_flag')

            ### Business related transilations to specific data. Need to be updated for each specific data set for scaling

            x = x.replace('partial paid', 'partial_paid')
            x = x.replace('not denied', 'paid')
            x = x.replace('not denied', 'partial_paid')
            x = x.replace('partially denied', 'partial_paid')
            x = x.replace('not paid', 'denied')
            x = x.replace('not paid completely ', 'partial_paid')
            x = x.replace('key accounts', "e&i_key_accts.")
            x = x.replace('non key accounts', "e&i_nat'l_accts.")
            x = x.replace('non key accounts', 'm&r')
            x = x.replace('non key accounts', 'private')
            x = x.replace('non key accounts', 'public')
            x = x.replace('non national accounts', 'e&i_key_accts.')
            x = x.replace('non national accounts', 'm&r')
            x = x.replace('non national accounts', 'private')
            x = x.replace('non national accounts', 'public')
            x = x.replace('non medicare', "e&i_nat'l_accts.")
            x = x.replace('non medicare', "e&i_key_accts.")
            x = x.replace('non m&r', "e&i_nat'l_accts.")
            x = x.replace('non m&r', "e&i_key_accts.")
            x = x.replace('non mnr', "e&i_nat'l_accts.")
            x = x.replace('non mnr', "e&i_key_accts.")
            x = x.replace(' eni ', " e&i_key_accts. ")
            x = x.replace(' eni ', " e&i_nat'l_accts. ")
            x = x.replace(' e&i ', " e&i_key_accts. ")
            x = x.replace(' e&i ', " e&i_nat'l_accts. ")
            x = x.replace('non e&i', 'm&r')
            x = x.replace('non eni', 'm&r')
            x = x.replace('mnr', 'm&r')
            x = x.replace('m&r', 'm&r')
            x = x.replace('medicare', 'm&r')
            x = x.replace(' eni ', ' unet ')
            x = x.replace(' e&i ', ' unet ')
            x = x.replace(' mnr ', ' cosmos ')
            x = x.replace(' m&r ', ' cosmos ')
            x = x.replace('medicare', 'cosmos')
            x = x.replace('line of business', 'line_of_business')
            x = x.replace('proc codes', 'proc_codes')
            x = x.replace('proc code', 'proc_code')
            x = x.replace('procedure codes', 'procedure_codes')
            x = x.replace('denial', 'denied')

            return x


        def tok_pos_tagger(x):

            x = nltk.word_tokenize(x)
            x = nltk.pos_tag(x)

            word = []
            tag = []

            for i in x:
                word.append(i[0])
                tag.append(i[1])

            df = pd.DataFrame({'word' : word, 'tag' : tag})

            lemmatizer = WordNetLemmatizer()

            for j in range(0,len(df['word'])):
                l2 = lemmatizer.lemmatize(df.loc[j,'word'])
                df.loc[j,'word']=l2

            return df


        ### Column name tagger

        def col_name_tagger(x):

            col_name_tags = []

            for i in x:
                k = col_dict.get(i)
                col_name_tags.append(k)

            return col_name_tags


        ### Column value match tagger

        def col_value_tagger(x):

            col_name_value_pair_tags = []

            for i in x:
                k = col_value_dict_unq.get(i)

                if i not in ("unknown_flag", "psm_grif", "date_num", "date_str"):   ### to be changed as per data set values
                    i_corrected = i.replace("_", " ")
                else:
                    i_corrected = i

                col_name_value_pair_tags.append([k, i_corrected])

            return col_name_value_pair_tags


        ### Converting to lower
        question = ques.lower()
        
        question=text2int(question)
        ### Extracting dates before replacing

        # dates = date_extract(question)

        # pre_post = pre_post_extract(question)

        ### Replacing dates -> Replacing essential values -> Tokenizing & tagging POS and converting to DF

        # df = tok_pos_tagger(replace_ess(date_replace(question)))
        df = tok_pos_tagger(replace_ess(question))

        ### Tagging Column names and adding column to the data frame

        col_name_tags = col_name_tagger(df['word'].tolist())

        df['col_name_tags'] = col_name_tags

        ### Tagging Column values and adding column & corrected filter value for to the data frame

        col_value_tags = col_value_tagger(df['word'].tolist())

        value_col_name = []
        filter_value = []

        for i in col_value_tags:
            value_col_name.append(i[0])
            filter_value.append(i[1])

        df['value_col_name'] = value_col_name
        df['filter_value'] = filter_value


        #df

        cont_col=['savings']
        date_col=['date_of_service','claim_received_date']
        cat_col=['client','platform','state','uhn_region','prov_spcl_catgy_desc','detection_method','claim_type','fwae_dipsosition','adjd_disposition_clm_lvl','npi','procedure_codes']
        id_col=['claim_number']

        ## main columns x,y
        c=[]
        y=[]
        x=[]
        for i in (df['col_name_tags'].tolist()):
            if i is not None:
                c.append(i)
                if i in cont_col:
                    y.append(i)
                elif i in cat_col:
                    x.append(i)
                elif i in id_col:
                    y.append(i)

        ## filter columns with filtervalue
        #filter={}

        filter={}
        for i in (df['value_col_name'].tolist()):
            if i is not None:
                filter.update( {i : df.loc[df['value_col_name']==i,'filter_value'].item()} )
                print i
            elif df['value_col_name'].unique() is None:
                filter='NA'
        
        def high_low(q):
            hh = 0
            for i in q:
                if i in ['top', 'most','first','upper']:
                    hh = 1
                    

                elif i in ['lowest', 'least','bottom','last']: 
                    hh = 2
            return hh
        hh=high_low(df['word'].tolist())
        #print k
        #print type(k)
        #if k==1:
         #   print True
        #abcd = df['word'].tolist()
        #print abcd
    ##number from top/low                                                    
        
        def numeric_extract(o):
            

            m = re.sub("[^0-9\s]", "", o).strip().split(" ")[0]
            if m!='':
                o=int(m)
            else:
                o=0

            return o
        num=numeric_extract(question)
    
        
      
        agg={'saving':'sum',
     'add':'sum',
     'sum':'sum',
     'addition':'sum',
     'summation':'sum',
    'max':'max',
    'maximum':'max',
    'minimum':'min',
    'min':'min',
     'average':'avg',
    'avg':'avg',
    'count':'count',
    'claims':'count',
    'claim':'count',
    'volume':'count'}
        aggregate_func=[]
        for i in (df['word'].tolist()):
            if i in agg.keys():
                aggregate_func.append(agg[i])
        print aggregate_func
    ## top low filtering
       

        #def amount_extract(x):
         #       if x.find("$") != -1:
          #          amount_present  = 1
           #         pos = x.find("$")
                #    x = x[(pos+1):(pos+8)]
                 #   x =  re.sub("[^0-9\s]", "", x)
                 #   x = x.strip()
                 #   if x.find(" ") != -1:
                 #       x = x.split(" ")[0]
                #elif x.find("dollar") != -1:
                 #   amount_present = 1
                  #  if x.find("dollar") != -1:
                   #     pos = x.find("dollar")
                   # if x.find("dollars") != -1:
                    #    pos = x.find("dollars")
                   # x = x[(pos-1):(pos-9)]
                   # x =  re.sub("[^0-9\s]", "", x)
                   # x = x.strip()
                   # if x.find(" ") != -1:
                    #    x = x.split(" ")[len(x.split(" ")) - 1]

                #x = int(x)

#                return x







        print x
        print y
        print filter
        

        ## plot req or not
 #       plot_req='False'
#        plot_words=['distribution','dist','chart','plot','draw','show','visualize','vizualization','vizualizations','distribute','distributions','distributed','dispersion','dispersed','disperse','allotment','allot','alloted','spread','disposition','dispositions','disbursement','disbursed','disburse','trend','trends','trending','variation','variations','change']
  #      for i in (df['word'].tolist()):
   #         if i in plot_words:
    #            plot_req='True'

#        print plot_req

        


        #reading
        data=pd.read_csv("*")
        data.columns = [p.lower() for p in data.columns]
        ## lower case+drop dup
        data=data.drop_duplicates()
        data=data.apply(lambda q: q.astype(str).str.lower())
        for i in cont_col:
            data[cont_col]=data[cont_col].astype(float)
        
        def filter_data(df):
            filter={}
            for i in (df['value_col_name'].tolist()):
                if i is not None:
                    filter.update( {i : list(df[df['value_col_name']==i].filter_value)} )
                elif df['value_col_name'].unique() is None:
                    filter='NA'
            return filter
        
        filter = filter_data(df)
        
        ## filtering data

        if filter !='NA':
            ch=[]
            for i in filter.keys():
                for k in filter[i]:
                    ch.append(k)

                data=data.loc[data[i].isin(ch)]


        #print data.shape
        ## filtering data

        #if filter !='NA':
         #   for i in filter.keys():
          #      data=data[data[i]==filter[i]]
        #print data.shape
        ##finding type of chart

        #if plot_req=='True':
         #   cardinality=[]
          #  for i in x:
           #     cardinality.append(data[i].nunique())
           # bool=cardinality >=6
            #if true in bool:
             #   type_of_chart='bar-chart'
           # else:
            #    type_of_chart='pie-chart'
        #elif plot_req=='False':
         #   type_of_chart='NA'
        


        
        ## plot req or not
        plot_req='False'
        plot_words=['distribution','dist','chart','plot','draw','visualize','vizualization','vizualizations','distribute','distributions','distributed','dispersion','dispersed','disperse','allotment','allot','alloted','spread','disposition','dispositions','disbursement','disbursed','disburse','trend','trends','trending','variation','variations','change']
        for i in (df['word'].tolist()):
            if i in plot_words:
                plot_req='True'
        type_of_chart=[]
        if plot_req=='True':
                for i in (df['word'].tolist()):
                    if i in (['trend','line','change']):
                        if aggregate_func[0] =="sum":
                            def fnctn(a,c):
                                #os.remove(filename)
                                df2 = pd.DataFrame(data.groupby(a)[c].sum())
                                df2=df2.astype(float)
                                plot1 = plt.plot(df2.index,df2[c])
                                plt.xticks(fontsize=10, rotation=30)
                                #plt.set_size_inches(8, 6)
                                plt.savefig("./static/q18.jpg")
                                plt.close()
                                plt.show()


                            fnctn(x,y)
                        elif aggregate_func[0] =="count":
                            def fnctn(a,c):
                                df2 = pd.DataFrame(data.groupby(a)[c].count())
                                df2=df2.astype(float)
                                plot1 = plt.plot(df2.index,df2[c])
                                plt.xticks(fontsize=10, rotation=30)
                                plt.savefig("./static/q18.jpg")
                                plt.show()
                                plt.show()

                            fnctn(x,y)

                        type_of_chart='line-chart'
                if type_of_chart != 'line-chart':
                    cardinality=[]
                    for i in x:
                        cardinality.append(data[i].nunique())
                    mm = [k for k in cardinality if k>6]
                    if len(mm)>0:
                        if aggregate_func[0] =="sum":
                            def fnctn(a,c):
                                df2 = pd.DataFrame(data.groupby(a)[c].sum())
                                df2=df2.astype(float)
                                #df2['col'] =df2.index
                                plot1 = sns.barplot(x=df2.index,y=df2.iloc[:,0])
                                plot1.set_xticklabels(plot1.get_xticklabels(), rotation=30, fontsize=6)
                                plot1.figure.savefig("./static/q18.jpg")
                                plt.close()
                                #plt.show()

                        elif aggregate_func[0] =="count":
                            def fnctn(a,c):
                                df2 = pd.DataFrame(data.groupby(a)[c].count())
                                df2=df2.astype(float)
                                plot1 = plt.bar(x=df2.index,y=df2.iloc[:,0])
                                plot1.set_xticklabels(plot1.get_xticklabels(), rotation=30, fontsize=6)
                                plot1.figure.savefig("./static/q18.jpg")
                                plt.close()
                                #plt.show()
                        fnctn(x,y)
                        type_of_chart='bar-chart'
                    else:
                        if aggregate_func[0] =="sum":
                            def fnctn(a,c):
                                df2 = pd.DataFrame(data.groupby(a)[c].sum())
                                df2=df2.astype(float)
                                plot = df2.plot.pie(y=c, figsize=(20, 20), autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2, textprops={'fontsize': 18})
                                print "xyz"
                                plt.show()
                                plot.figure.savefig("./static/q18.jpg")
                                plt.close()
                            fnctn(x,y)          
                        elif aggregate_func[0] =="count":
                            def fnctn(a,c):
                                df2 = pd.DataFrame(data.groupby(a)[c].count())
                                df2=df2.astype(float)
                                plot = df2.plot.pie(y=c, figsize=(20, 20), autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2, textprops={'fontsize': 18})
                                print "xyz"
                                plt.show()
                                plot.figure.savefig("./static/q18.jpg")
                                plt.close()
                            fnctn(x,y)  
                        type_of_chart='pie-chart'
        elif plot_req=='False':
            type_of_chart='NA'
        
        def last(a,c):
            #k = 1
            if hh==1:
                if aggregate_func[0] =="sum":
                    print hh
                    df2 = pd.DataFrame(data.groupby(a)[c].sum())
                    #df2['col']=df2.index
                    

                    print df2

                    df2=df2.sort_values(by=c, ascending=False)
                    df2=df2[:num]
                    #df2=df2.head(num)
                    df2=df2.astype(float)
                    plot = df2.plot.pie(y=c, figsize=(20, 20), autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2, textprops={'fontsize': 18})
                    print "xyz"
                    plt.show()
                    plot.figure.savefig("./static/q18.jpg")
                    plt.close()
                if aggregate_func[0] =="count":
                    df2 = pd.DataFrame(data.groupby(a)[c].count())
                    #df2['col']=df2.index
                    
                    print df2

                    df2=df2.sort_values(by=c, ascending=False)
                    df2=df2[:num]
                    #df2=df2.head(num)
                    df2=df2.astype(float)
                    plot = df2.plot.pie(y=c, figsize=(20, 20), autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2, textprops={'fontsize': 18})
                    print "xyz"
                    plt.show()
                    plot.figure.savefig("./static/q18.jpg")
                    plt.close()
            if hh==2:
                if aggregate_func[0] =="sum":
                    df2 = pd.DataFrame(data.groupby(a)[c].sum())
                    #df2['col']=df2.index
                    
                    print df2

                    df2=df2.sort_values(by=c, ascending=True)
                    df2=df2[:num]
                    #df2=df2.head(num)
                    df2=df2.astype(float)
                    plot = df2.plot.pie(y=c, figsize=(20, 20), autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2, textprops={'fontsize': 18})
                    print "xyz"
                    plt.show()
                    plot.figure.savefig("./static/q18.jpg")
                    plt.close()
                if aggregate_func[0] =="count":
                    df2 = pd.DataFrame(data.groupby(a)[c].count())
                    #df2['col']=df2.index
                    
                    print df2

                    df2=df2.sort_values(by=c, ascending=True)
                    df2=df2[:num]
                    #df2=df2.head(num)
                    df2=df2.astype(float)
                    plot = df2.plot.pie(y=c, figsize=(20, 20), autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2, textprops={'fontsize': 18})
                    print "xyz"
                    plt.show()
                    plot.figure.savefig("./static/q18.jpg")
                    plt.close()

        last(x,y)  
##finding type of chart &plotting it in it

    

    #type_of_chart
        
       
        #if aggregate_func[0] =="sum":
         #   def fnctn(a,c):
        #for i in range(len(data)):
     #   filtering = data.loc[data[a].isin(b)]
           #     df2 = pd.DataFrame(data.groupby(a)[c].count())
          #      df2=df2.astype(float)
           #     plot = df2.plot.pie(y=c, figsize=(20, 20))
            #    plot.figure.savefig("./static/q18.jpg")
            #fnctn(x,y)
        #elif aggregate_func[0]=="count":
         #   def fnctn(a,c):
        #for i in range(len(data)):
     #   filtering = data.loc[data[a].isin(b)]
          #      df2 = pd.DataFrame(data.groupby(a)[c].count())
           #     df2=df2.astype(float)
            #    plot = df2.plot.pie(y=c, figsize=(20, 20))
             #   plot.figure.savefig("./static/q18.jpg")
           # fnctn(x,y)#plt.show()
        #elif aggregate_func[0]=="avg":
        #    def fnctn(a,c):
       # for i in range(len(data)):
     #   filtering = data.loc[data[a].isin(b)]
            #    df2 = pd.DataFrame(data.groupby(a)[c].count())
             #   df2=df2.astype(float)
              #  plot = df2.plot.pie(y=c, figsize=(20, 20))
               # plot.figure.savefig("./static/q18.jpg")
        #    fnctn(x,y)#plt.show()
        #elif aggregate_func[0]=="max":
       #     def fnctn(a,c):
        #for i in range(len(data)):
     #   filtering = data.loc[data[a].isin(b)]
       #         df2 = pd.DataFrame(data.groupby(a)[c].max())
      #          df2=df2.astype(float)
     #           plot = df2.plot.pie(y=c, figsize=(20, 20))
    #            plot.figure.savefig("./static/q18.jpg")#plt.show()
        
   #         fnctn(x,y)
    url ="./static/q18.jpg"
    return render_template('home.html', title='Home', form=form,text="abc",url=url)
    #return(dbname)
#print(dbname)

#print (dbname)
@app.route("/test2", methods=['GET', 'POST'])
def home2():
    global dbname
    form = RegistrationForm()
    if request.method == "POST":
        unm = request.form.get("username", None)
    return render_template('home.html', title='Home', form=form)    

if __name__ == '__main__':
   app.run(debug= True,host = "*",port = *)