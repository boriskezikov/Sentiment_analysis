import sys
import decimal
import codecs
import re
import time
import collections
#nltk.download("stopwords")
from nltk.corpus import stopwords
import pymorphy2
import itertools
import datetime
import numpy as np
import matplotlib.pyplot as plt


russian_stopwords = stopwords.words("russian")
stoptags = ['NUMR', 'NPRO', 'PRED', 'CONJ', 'PRCL', 'INTJ', 'None', 'PREP', 'GRND', 'LATN', 'PNCT']
start = time.time()

main_data = r'C:\Users\Boris\Desktop\data.txt'
tvits = r"E:\data\first_parsed.txt"

morphalizer = pymorphy2.MorphAnalyzer()

error_log = open(r"E:\data\error_log.txt",'w',encoding='utf-8')


class Computing:

    def __init__(self):
        self.twit_count = 0  # Количество твитов
        self.estimation_table = {}  # таблица со словами и их оценкой (-1,0,1)
        self.frequency_table = {}  # частота встрачемости слов (слов - частота)
        self.words_in_twits = {}    # таблица хранящая твит как ключ и тональные слова как значение

        self.twit_tonality_rule1 = {}  # хранит твит и его тональность согласно 1 правилу
        self.twit_tonality_rule2 = {}  # хранит твит и его тональность согласно 2 правилу
        self.twit_tonality_rule3 = {}  # хранит твит и его тональность соласно 3 правилу
        self.twit_tonality_rule4 = {}  # хранит твит и его тональность соласно 4 правилу

        self.adjectives_store = {}  # хранит прилагательные и их частоту встречаемости
        self.adjective_marks = {}  # хранит прилагательные и его оценку

        self.mark_counts_1rule = []  # хранит кол во твитов по 1 правилу 1-pos 2 neg 3 neu
        self.mark_counts_2rule = []  # хранит кол во твитов по 2 правилу 1-pos 2 neg 3 neu
        self.mark_counts_3rule = []  # хранит кол во твитов по 3 правилу 1-pos 2 neg 3 neu
        self.mark_counts_4rule = []  # хранит кол во твитов по 4 правилу 1-pos 2 neg 3 neu

        self.top_adj_pos = {}  # Хранит топ положительных прил
        self.top_adj_neg = {}  # Хранит топ отриц прил
        self.twits_with_dates = {}  # твит и час

        # Хранят данные по распределениям классов во времени в зависимости от выбора функции оценивания
        self.pos_distribution1 = []
        self.neg_distribution1 = []
        self.neu_distribution1 = []
        self._time_distribution1 = []
        self.current_twits_sum1 = []

        self.pos_distribution2 = []
        self.neg_distribution2 = []
        self.neu_distribution2 = []
        self._time_distribution2 = []
        self.current_twits_sum2 = []

        self.pos_distribution3 = []
        self.neg_distribution3 = []
        self.neu_distribution3 = []
        self._time_distribution3 = []
        self.current_twits_sum3 = []

        self.pos_distribution4 = []
        self.neg_distribution4 = []
        self.neu_distribution4 = []
        self._time_distribution4 = []
        self.current_twits_sum4 = []

    def cleaner(self):
        with codecs.open(r"E:\data\data.txt", 'r', encoding="utf-8") as readable:
            with open(r"E:\data\first_parsed_tvits.txt", 'w', encoding="utf-8") as writable:
                for string in readable:

                    string = re.sub("#[ ]*[A-Za-zA-Яа-яё0-9]*",'',string) # tags
                    string = re.sub("(?:pic.twitter.com/|https://|http://|.twitter.com/|.ru|.com|.org)[^ \n]*", '', string)  # references
                    string = re.sub("@[ ]*[^ \n]*", '', string)  # @user
                    string = re.sub("[!.,@$%^&*()\-_=+\"№;?/`:<>{}\[\]']", '', string)
                    if not string.isspace():
                        data = string.split()[1]
                        data = datetime.datetime(1, 1, 1, int(data[0:2]), int(data[2:4]))
                    string = re.sub("\d", '', string)
                    if not string.isspace():
                        zir = re.sub('\n','',string)
                        zir = re.sub('\ufeff','',string)
                        self.twits_with_dates[data] = zir
                        writable.write(string)
        second_parsed_lemmatized = open(r"E:\data\second_parsed_lemmatized.txt", 'w', encoding="utf-8")

        with codecs.open(r"E:\data\first_parsed_tvits.txt", 'r', encoding="utf-8") as twiter:

            for line in twiter:
                self.twit_count += 1
                for string in line.split():
                    if (string not in russian_stopwords) or (4 > len(string) > 15):
                        if string not in self.frequency_table:
                            new_word = morphalizer.parse(string)[0]
                            TAG = str(new_word.tag.POS)
                            if TAG in stoptags:
                                #print("break")
                                break
                            new_word = new_word.normal_form.lower()
                            if (new_word not in russian_stopwords) and (15 > len(new_word) > 4):  # удаляем слова из stopwords а так же все слова длиной короче 4 букв
                                if new_word not in self.frequency_table:
                                    self.frequency_table[new_word] = 1  # если элемент встретился 1 раз, добавляем его в таблицу
                                    second_parsed_lemmatized.write(new_word + '\n')
                                    #print("Записано новое слово")

                                else:
                                    self.frequency_table[new_word] += 1
                                    #print("Значение слова увеличилось на 1")
                            else:
                                continue

            second_parsed_lemmatized.close()
            print("Функция cleaner работает", time.time() - start, "секунд\n")
            return 0

    def frequency(self):
        start_time = time.time()
        frequency = open(r"E:\data\frequency.txt", 'w', encoding="utf-8")
        for i in sorted(self.frequency_table, key=self.frequency_table.get):
            frequency.write(str(self.frequency_table[i]) + ' - ' + str(i)+' - ' + str(round((((self.frequency_table[i]) / self.twit_count) * 100),5))+'%' + '\n')
        frequency.close()
        print("Функция frequency работает", time.time() - start_time, "секунд\n")
        return 0

    def declarator(self):
        start_time = time.time()
        parsed_twits = []

        estimator = open(r"E:\data\estimation.txt", 'r', encoding='utf-8')
        self.estimation_table = {line.split()[0]: line.split()[1] for line in estimator}
        estimator.close()

        twits = open(r"E:\data\first_parsed_tvits.txt", 'r', encoding="utf-8")
        for twit in twits:
            twit = re.sub('\n', '', twit)
            twit = re.sub('\ufeff', '', twit)
            parsed_twits.append(twit)
        twits.close()

        '''Создаем список слов из файла с лемматизированными словами'''
        info = open(r"E:\data\second_parsed_lemmatized.txt", 'r', encoding='utf-8')
        words_lemmatized = []
        for word in info:
            words_lemmatized.append(re.sub('\n','',word))
        info.close()

        '''Создаем словарь {key - Твит, value - список слов}'''
        for twit in parsed_twits:
            for word in twit.split():
                check_word = morphalizer.parse(word)[0].normal_form.lower()
                if check_word in words_lemmatized:
                    if self.words_in_twits.get(twit):
                        # append the new number to the existing array at this slot
                        self.words_in_twits[twit].append(check_word)
                    else:
                        # create a new array in this slot
                        self.words_in_twits[twit] = [check_word]
        self.twit_count = len(self.words_in_twits)


        print("Функция declartor работает", time.time() - start_time, "секунд\n")
        return 0

    def rules_maker(self):
        start_time = time.time()

        def first_rule():
            for twit in self.words_in_twits:
                twit_mark = 0
                try:
                    for word in self.words_in_twits.get(twit):
                        twit_mark += int(self.estimation_table.get(word))
                    self.twit_tonality_rule1[twit] = twit_mark
                except TypeError:
                    error_log.writelines("Ошибка в твите: {} \nПрисвоено значение 0".format(twit))
                    self.twit_tonality_rule1[twit] = 0

            neg = 0
            pos = 0
            neu = 0
            for twit in self.twit_tonality_rule1:
                if -1 <= self.twit_tonality_rule1[twit] <= 0:
                    neu += 1
                elif self.twit_tonality_rule1[twit] < -1:
                    neg += 1
                elif self.twit_tonality_rule1[twit] > 0:
                    pos += 1
                else:
                    neu += 1
            self.mark_counts_1rule.append(pos)
            self.mark_counts_1rule.append(neg)
            self.mark_counts_1rule.append(neu)

            positive = "Positive - {0} - {1}%\n".format(pos, round(pos/self.twit_count*100, 3))
            negative = "Negative - {0} - {1}%\n".format(neg, round(neg/self.twit_count*100, 3))
            neutral = "Neutral - {0} - {1}%\n".format(neu, round(neu/self.twit_count*100, 3))

            with open(r"E:\data\classification.txt", 'w', encoding="utf-8") as classification:
                classification.write("Classification by mark summary\n")
                classification.write(str(positive))
                classification.write(str(negative))
                classification.write(str(neutral))
                classification.write("______________________________________________________")


        def second_rule():
            for twit in self.words_in_twits:
                sum_words = len(twit.split())
                sum_pos = 0
                sum_neg = 0
                sum_neu = 0
                try:
                    word_list = self.words_in_twits.get(twit)
                    for word in word_list:
                        word_estimation = int(self.estimation_table.get(word))
                        if word_estimation == 0:
                            sum_neu += 1
                        elif word_estimation == -1:
                            sum_neg += 1
                        elif word_estimation == 1:
                            sum_pos += 1
                        else:
                            sum_neu += 1

                    pos_percentage = sum_pos / sum_words * 100
                    neg_percentage = sum_neg / sum_words * 100
                    neu_percentage = sum_neu / sum_words * 100

                    if max(pos_percentage, neg_percentage, neu_percentage) == pos_percentage:
                        self.twit_tonality_rule2[twit] = 1
                    elif max(pos_percentage, neg_percentage, neu_percentage) == neg_percentage:
                        self.twit_tonality_rule2[twit] = -1
                    elif max(pos_percentage, neg_percentage, neu_percentage) == neu_percentage:
                        self.twit_tonality_rule2[twit] = 0
                except:
                    error_log.writelines("Ошибка в твите: {} \nПрисвоено значение 0".format(twit))
                    self.twit_tonality_rule2[twit] = 0

            neg = 0
            pos = 0
            neu = 0
            for twit in self.twit_tonality_rule2:
                if self.twit_tonality_rule2[twit] == 0:
                    neu += 1
                elif self.twit_tonality_rule2[twit] == -1:
                    neg += 1
                elif self.twit_tonality_rule2[twit] == 1:
                    pos += 1
                else:
                    neu += 1

            self.mark_counts_2rule.append(pos)
            self.mark_counts_2rule.append(neg)
            self.mark_counts_2rule.append(neu)

            positive = "Positive - {0} - {1}%\n".format(pos, round(pos / self.twit_count * 100, 4))
            negative = "Negative - {0} - {1}%\n".format(neg, round(neg / self.twit_count * 100, 4))
            neutral = "Neutral - {0} - {1}%\n".format(neu, round(neu / self.twit_count * 100, 4))

            with open(r"E:\data\classification.txt", 'a', encoding="utf-8") as classification:
                classification.write("\n")
                classification.write("Classification by word types\n")
                classification.write(str(positive))
                classification.write(str(negative))
                classification.write(str(neutral))
                classification.write("______________________________________________________\n")
           # print("\nClassification by word types\n")
           # print(positive)
           # print(negative)
            #print(neutral)

        def third_rule():

            for twit in self.words_in_twits:
                adj_weight = 0
                adv_weight = 0
                noun_weight = 0
                verb_weight = 0
                word_list = self.words_in_twits.get(twit)
                for word in word_list:
                    word = morphalizer.parse(word)[0]
                    TAG = str(word.tag.POS)
                    word = word.normal_form.lower()
                    try:
                        if TAG == "ADJF" or TAG =="ADJS":
                            par = int(self.estimation_table.get(word))
                            adj_weight += par

                        elif TAG == "ADVB":
                            par = int(self.estimation_table.get(word))
                            adv_weight += par

                        elif TAG =="NOUN":
                            par = int(self.estimation_table.get(word))
                            noun_weight += par

                        elif TAG == "VERB" or TAG == "INFN":
                            par = int(self.estimation_table.get(word))
                            verb_weight += par

                    except BaseException as exception:
                        error_log.write(str("\nОшибка в применении 3 правила: "))
                        error_log.write(str(exception))
                        continue

                twit_mark = 0.1*noun_weight + 0.3 * adv_weight + 0.4*adj_weight + 0.2*verb_weight
                self.twit_tonality_rule3[twit] = twit_mark
            neg = 0
            pos = 0
            neu = 0
            for twit in self.twit_tonality_rule3:
                if self.twit_tonality_rule3.get(twit) > 0:
                    pos += 1
                elif self.twit_tonality_rule3.get(twit) < 0:
                    neg += 1
                else:
                    neu += 1

            self.mark_counts_3rule.append(pos)
            self.mark_counts_3rule.append(neg)
            self.mark_counts_3rule.append(neu)
            positive = "Positive - {0} - {1}%\n".format(pos, round(pos / self.twit_count * 100, 3))
            negative = "Negative - {0} - {1}%\n".format(neg, round(neg / self.twit_count * 100, 3))
            neutral = "Neutral - {0} - {1}%\n".format(neu, round(neu / self.twit_count * 100, 3))

            with open(r"E:\data\classification.txt", 'a', encoding="utf-8") as classification:
                classification.write("\nClassification by speech part diversification\n")
                classification.write(str(positive))
                classification.write(str(negative))
                classification.write(str(neutral))
                classification.write("______________________________________________________\n")


        def forth_rule():
            for twit in self.words_in_twits:
                pos_weight = 0
                neg_weight = 0
                neu_weight = 0
                try:
                    for word in self.words_in_twits.get(twit):
                        par = int(self.estimation_table.get(word))
                        if par == 1:
                            pos_weight += 1

                        elif par == -1:
                            neg_weight += 1

                        elif par == 0:
                            neu_weight += 1

                    twit_mark = 0.4*pos_weight + 0.4*neg_weight + 0.2*neu_weight

                    self.twit_tonality_rule4[twit] = twit_mark
                except TypeError:
                    self.twit_tonality_rule4[twit] = 0

            neg = 0
            pos = 0
            neu = 0
            for twit in self.twit_tonality_rule4:
                if 2 <= self.twit_tonality_rule4[twit] <= 6:
                    neu += 1
                elif self.twit_tonality_rule4[twit] < 2:
                    neg += 1
                elif self.twit_tonality_rule4[twit] > 6:
                    pos += 1
                else:
                    neu += 1



            self.mark_counts_4rule.append(pos)
            self.mark_counts_4rule.append(neg)
            self.mark_counts_4rule.append(neu)

            positive = "Positive - {0} - {1}%\n".format(pos, round(pos/self.twit_count*100, 3))
            negative = "Negative - {0} - {1}%\n".format(neg, round(neg/self.twit_count*100, 3))
            neutral = "Neutral - {0} - {1}%\n".format(neu, round(neu/self.twit_count*100, 3))

            with open(r"E:\data\classification.txt", 'a', encoding="utf-8") as classification:
                classification.write("Classification by average mark summary\n")
                classification.write(str(positive))
                classification.write(str(negative))
                classification.write(str(neutral))
                classification.write("______________________________________________________")

        first_rule()
        second_rule()
        third_rule()
        forth_rule()
        print("\nФункция rules_maker работает ", time.time() - start_time, 'секунд.\n\r')

    def adjectiver(self):
        start_time = time.time()
        for twit in self.words_in_twits:
                for word in self.words_in_twits.get(twit):
                        word = morphalizer.parse(word)[0]
                        TAG = str(word.tag.POS)
                        if TAG == "ADJF" or TAG == "ADJS":
                            word = word.normal_form.lower()
                            if word not in self.adjectives_store:
                                self.adjectives_store[word] = 1
                            else:
                                self.adjectives_store[word] += 1

        with open(r"E:\data\adjectivefrequency.txt", 'w', encoding='utf-8') as f:
            for note in sorted(self.adjectives_store.items()):
                f.write(str(note))
                f.write('\n')


        adj_pos = {}
        adj_neg = {}
        ordered = dict(sorted(self.adjectives_store.items(), key=lambda x: x[1], reverse=True))  # СКОРОСТЬ!!!!!

        for adj in self.adjectives_store:
            mark = self.estimation_table.get(adj)
            if mark == str(1):
                adj_pos[adj] = self.estimation_table.get(adj)
            elif mark == str(-1):
                adj_neg[adj] = self.estimation_table.get(adj)
        j = 0

        for adj in ordered:
            if j != 5:
                if adj in adj_pos:
                    self.top_adj_pos[adj] = self.adjectives_store.get(adj)
                    j += 1
                else:
                    continue
            else:
                break

        j = 0
        for adj in ordered:
            if j != 5:
                if adj in adj_neg:
                    self.top_adj_neg[adj] = self.adjectives_store.get(adj)
                    j += 1
                else:
                    continue
            else:
                break


        with open(r"E:\data\adjectives.txt", 'w', encoding='utf-8') as adjective_file:

            adjective_file.write("TOP-5 POSITIVE\n")
            for i in self.top_adj_pos:
                adjective_file.write(str(i + ' ' + str(self.top_adj_pos.get(i)) + ' - ' +
                                     str(round(((self.top_adj_pos.get(i)/self.twit_count)*100), 3)) + '%' + '\n'))
            adjective_file.write("\n___________________________________________\n")

            adjective_file.write("\nTOP-5 NEGATIVE\n")

            for i in self.top_adj_neg:
                adjective_file.write(str(i + ' ' + str(self.top_adj_neg.get(i)) + ' - ' +
                                     str(round((self.top_adj_neg.get(i)/self.twit_count*100),3)) + '%' + '\n'))
            adjective_file.write("\n___________________________________________\n")

        print("\nФункция adjectiver работает ", time.time()-start_time," секунд.\n")  #

    def time_distribution(self):

        starter = time.time()
        new_twits_dates = {}
        writable = open(r"E:\data\hours.txt", 'w', encoding="utf-8")
        for i in self.twits_with_dates.keys():
            string = re.sub('\n', '', self.twits_with_dates.get(i))
            new_twits_dates[i] = string
        def rool1():
            time_gap = sorted(self.twits_with_dates)[0] + datetime.timedelta(minutes=30)
            start_time = sorted(self.twits_with_dates)[0]

            start_time_r = str(start_time)[11:]
            writable.write("\n______First rule estimation _______\n")
            while time_gap <= sorted(self.twits_with_dates)[len(sorted(self.twits_with_dates))-1]:
                pos = 0
                neg = 0
                neu = 0
                for i in sorted(new_twits_dates.keys()):
                    if i <= time_gap:
                        twit = new_twits_dates.get(i)
                        if twit in self.twit_tonality_rule1:
                            key = self.twit_tonality_rule1.get(twit)
                            if key:
                                if key > 0:
                                    pos += 1
                                elif key < -1:
                                    neg += 1
                                else:
                                    neu += 1
                            else:
                                continue
                        else:
                            continue
                count = pos+neg+neu
                positive = round((pos/count*100), 2)
                negative = round((neg/count*100), 2)
                neutral = round((neu/count*100), 2)
                time_to_write = str(time_gap)[11:20]

                to_write = str("\n{0} - {1}:{2}        {3} / {4} / {5}\n").format(start_time_r,time_to_write,count,positive,negative,neutral)

                writable.write(to_write)
                self.pos_distribution1.append(pos)
                self.neg_distribution1.append(neg)
                self.neu_distribution1.append(neu)
                self._time_distribution1.append(time_gap)
                self.current_twits_sum1.append(count)
                time_gap += datetime.timedelta(minutes=10)

            """______________Второе правило_________________"""
        def rool2():
            time_gap = sorted(self.twits_with_dates)[0] + datetime.timedelta(minutes=30)
            start_time = sorted(self.twits_with_dates)[0]
            start_time_r = str(start_time)[11:]
            writable.write("\n______Second rule estimation _______\n")
            while time_gap <= sorted(self.twits_with_dates)[len(sorted(self.twits_with_dates)) - 1]:
                pos = 0
                neg = 0
                neu = 0
                for i in sorted(new_twits_dates.keys()):

                    if i <= time_gap:
                        twit = new_twits_dates.get(i)
                        if twit in self.twit_tonality_rule2:
                            key = self.twit_tonality_rule2.get(twit)
                            if key:
                                if key == 1:
                                    pos += 1
                                elif key == -1:
                                    neg += 1
                                else:
                                    neu += 1
                            else:
                                continue
                        else:
                            continue
                count = pos + neg + neu
                positive = round((pos / count * 100), 2)
                negative = round((neg / count * 100), 2)
                neutral = round((neu / count * 100), 2)
                time_to_write = str(time_gap)[11:20]

                to_write = str("\n{0} - {1}:{2}        {3} / {4} / {5}\n").format(start_time_r, time_to_write,
                                                                                  count, positive, negative,
                                                                                  neutral)

                writable.write(to_write)
                self.pos_distribution2.append(pos)
                self.neg_distribution2.append(neg)
                self.neu_distribution2.append(neu)
                self._time_distribution2.append(time_gap)
                self.current_twits_sum2.append(count)
                time_gap += datetime.timedelta(minutes=10)

            """______________Третье правило_________________"""
        def rool3():
            time_gap = sorted(self.twits_with_dates)[0] + datetime.timedelta(minutes=30)
            start_time = sorted(self.twits_with_dates)[0]
            start_time_r = str(start_time)[11:]
            writable.write("\n______Third rule estimation _______\n")
            while time_gap <= sorted(self.twits_with_dates)[len(sorted(self.twits_with_dates)) - 1]:
                pos = 0
                neg = 0
                neu = 0
                for i in sorted(new_twits_dates.keys()):
                    if i <= time_gap:
                        twit = new_twits_dates.get(i)
                        if twit in self.twit_tonality_rule2:
                            key = self.twit_tonality_rule2.get(twit)
                            if key:
                                if key > 0:
                                    pos += 1
                                elif key < 0:
                                    neg += 1
                                else:
                                    neu += 1
                            else:
                                continue
                        else:
                            continue
                count = pos + neg + neu
                positive = round((pos / count * 100), 2)
                negative = round((neg / count * 100), 2)
                neutral = round((neu / count * 100), 2)
                time_to_write = str(time_gap)[11:20]

                to_write = str("\n{0} - {1}:{2}        {3} / {4} / {5}\n").format(start_time_r, time_to_write,
                                                                                  count, positive, negative,
                                                                                  neutral)

                writable.write(to_write)
                self.pos_distribution3.append(pos)
                self.neg_distribution3.append(neg)
                self.neu_distribution3.append(neu)
                self._time_distribution3.append(time_gap)
                self.current_twits_sum3.append(count)

                time_gap += datetime.timedelta(minutes=10)
        def rool4():
            time_gap = sorted(self.twits_with_dates)[0] + datetime.timedelta(minutes=30)
            start_time = sorted(self.twits_with_dates)[0]
            start_time_r = str(start_time)[11:]
            writable.write("\n______Forth rule estimation _______\n")
            while time_gap <= sorted(self.twits_with_dates)[len(sorted(self.twits_with_dates)) - 1]:
                pos = 0
                neg = 0
                neu = 0
                for i in sorted(new_twits_dates.keys()):
                    if i <= time_gap:
                        twit = new_twits_dates.get(i)
                        if twit in self.twit_tonality_rule2:
                            key = self.twit_tonality_rule2.get(twit)
                            if key:
                                if key >= 6:
                                    pos += 1
                                elif key <= 2:
                                    neg += 1
                                else:
                                    neu += 1
                            else:
                                continue
                        else:
                            continue
                count = pos + neg + neu
                positive = round((pos / count * 100), 2)
                negative = round((neg / count * 100), 2)
                neutral = round((neu / count * 100), 2)
                time_to_write = str(time_gap)[11:20]

                to_write = str("\n{0} - {1}:{2}        {3} / {4} / {5}\n").format(start_time_r, time_to_write,
                                                                                  count, positive, negative,
                                                                                  neutral)

                writable.write(to_write)
                self.pos_distribution4.append(pos)
                self.neg_distribution4.append(neg)
                self.neu_distribution4.append(neu)
                self._time_distribution4.append(time_gap)
                self.current_twits_sum4.append(count)

                time_gap += datetime.timedelta(minutes=10)
        rool1()
        rool2()
        rool3()
        rool4()
        print("\nФункция time_distribution  работает ", time.time() - starter, 'секунд.\n\r')

    def bar_charts(self):
        start_time = time.time()
        plt.rcdefaults()

        objects = ('Good', 'Bad', 'Neutral')

        x = np.arange(len(objects))

        height = [self.mark_counts_1rule[0], self.mark_counts_1rule[1], self.mark_counts_1rule[2]]

        plt.bar(x, height, align='center', alpha=0.5)
        plt.xticks(x, objects)
        plt.ylabel('Count')
        plt.title('Classification by sum')
        plt.savefig(r'E:\data\1 bar chart.png')


        height = [self.mark_counts_2rule[0], self.mark_counts_2rule[1], self.mark_counts_2rule[2]]

        plt.bar(x, height, align='center', alpha=0.5)
        plt.xticks(x, objects)
        plt.ylabel('Count')
        plt.title('Classification by dominance')
        plt.savefig(r'E:\data\2 bar chart.png')


        height = [self.mark_counts_3rule[0], self.mark_counts_3rule[1], self.mark_counts_3rule[2]]

        plt.bar(x, height, align='center', alpha=0.5)
        plt.xticks(x, objects)
        plt.ylabel('Count')
        plt.title('Classification by speech part diversification')
        plt.savefig(r'E:\data\3 bar chart.png')


        height = [self.mark_counts_4rule[0], self.mark_counts_4rule[1], self.mark_counts_4rule[2]]

        plt.bar(x, height, align='center', alpha=0.5)
        plt.xticks(x, objects)
        plt.ylabel('Count')
        plt.title('Classification by average mark')
        plt.savefig(r'E:\data\4 bar chart.png')
        plt.show()


        x = ('1st', '2nd', '3d', '4th', '5th')
        x_pos = np.arange(len(x))
        # Толщина столбца
        width = 0.2
        # Значения по Oy
        y_positive = [x for x in self.top_adj_pos.values()]
        y_negative = [x for x in self.top_adj_neg.values()]
        # Графики
        plt.bar(x_pos, y_positive, width=width, label='Positive', color='y')
        plt.bar(x_pos + width, y_negative, width=width, label='Negative', color='g')
        plt.ylabel('Amount')
        plt.xticks(x_pos + width / 2, x)
        plt.title('Top-5 Positive/Negative Adjectives')
        plt.legend()
        plt.savefig(r'E:\data\Adjectives bar chart.png')






        # Создаем области
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(27, 6))
        # Значения по Ox
        x = [key for key in self._time_distribution1]
        x_pos = np.arange(len(x))
        # Первый график
        ax1.plot(x_pos, self.pos_distribution1, 'g.-', linewidth=1, markersize=7, label='N_pos')
        ax1.plot(x_pos, self.neg_distribution1, 'b.--', linewidth=1, markersize=7, label='N_neg')
        ax1.plot(x_pos, self.neu_distribution1, 'r.-.', linewidth=1, markersize=7, label='N_neu')
        ax1.set_xticks([])
        ax1.grid(True)
        ax1.legend()
        ax1.set_ylabel('Fraction')
        # столбцы
        ax2.stem(x_pos, self.current_twits_sum1, basefmt=" ")
        # координатн ось
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(x)
        ax2.grid(True)
        ax2.set_ylabel('Number of tweets')
        fig.suptitle('Distribution of tweets classes in time rule 1', fontsize=16, y=1.0)
        fig.savefig(r'E:\data\Time distribution1.png')


        """2 правило """
        # Создаем области
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(27, 6))
        # Значения по Ox
        x = [key for key in self._time_distribution2]
        x_pos = np.arange(len(x))
        # Первый график
        ax1.plot(x_pos, self.pos_distribution2, 'g.-', linewidth=1, markersize=7, label='N_pos')
        ax1.plot(x_pos, self.neg_distribution2, 'b.--', linewidth=1, markersize=7, label='N_neg')
        ax1.plot(x_pos, self.neu_distribution2, 'r.-.', linewidth=1, markersize=7, label='N_neu')
        ax1.set_xticks([])
        ax1.grid(True)
        ax1.legend()
        ax1.set_ylabel('Fraction')
        # столбцы
        ax2.stem(x_pos, self.current_twits_sum2, basefmt=" ")
        # координатн ось
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(x)
        ax2.grid(True)
        ax2.set_ylabel('Number of tweets')
        fig.suptitle('Distribution of tweets classes in time rule 2', fontsize=16, y=1.0)
        fig.savefig(r'E:\data\Time distribution2.png')


        """3 правило """
        # Создаем области
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(27, 6))
        # Значения по Ox
        x = [key for key in self._time_distribution3]
        x_pos = np.arange(len(x))
        # Первый график
        ax1.plot(x_pos, self.pos_distribution3, 'g.-', linewidth=1, markersize=7, label='N_pos')
        ax1.plot(x_pos, self.neg_distribution3, 'b.--', linewidth=1, markersize=7, label='N_neg')
        ax1.plot(x_pos, self.neu_distribution3, 'r.-.', linewidth=1, markersize=7, label='N_neu')
        ax1.set_xticks([])
        ax1.grid(True)
        ax1.legend()
        ax1.set_ylabel('Fraction')
        # столбцы
        ax2.stem(x_pos, self.current_twits_sum3, basefmt=" ")
        # координатн ось
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(x)
        ax2.grid(True)
        ax2.set_ylabel('Number of tweets')
        fig.suptitle('Distribution of tweets classes in time rule 3', fontsize=16, y=1.0)
        fig.savefig(r'E:\data\Time distribution3.png')


        """4 правило """
        # Создаем области
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(27, 6))
        # Значения по Ox
        x = [key for key in self._time_distribution4]
        x_pos = np.arange(len(x))
        # Первый график
        ax1.plot(x_pos, self.pos_distribution4, 'g.-', linewidth=1, markersize=7, label='N_pos')
        ax1.plot(x_pos, self.neg_distribution4, 'b.--', linewidth=1, markersize=7, label='N_neg')
        ax1.plot(x_pos, self.neu_distribution4, 'r.-.', linewidth=1, markersize=7, label='N_neu')
        ax1.set_xticks([])
        ax1.grid(True)
        ax1.legend()
        ax1.set_ylabel('Fraction')
        # столбцы
        ax2.stem(x_pos, self.current_twits_sum4, basefmt=" ")
        # координатн ось
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(x)
        ax2.grid(True)
        ax2.set_ylabel('Number of tweets')
        fig.suptitle('Distribution of tweets classes in time rule 4', fontsize=16, y=1.0)
        fig.savefig(r'E:\data\Time distribution4.png')
        print("\nГрафики строятся  за ", time.time() - start_time," секунд.\n")

if __name__ == "__main__":

    start_program = time.time()

    parser = Computing()

    parser.cleaner()

    parser.frequency()

    parser.declarator()

    parser.rules_maker()

    parser.adjectiver()

    parser.time_distribution()

    parser.bar_charts()

    error_log.close()

    print("\nПрограмма работает", time.time() - start_program, "секунд\n")






































