# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []
from typing import Any, Text, Dict, List, Union
# from dotenv import load_dotenv

import numpy as np
import pandas as pd
import pickle,datetime,uuid

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.forms import FormAction
from rasa_sdk.events import AllSlotsReset, SlotSet

from textblob import Word
import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re, string
from textblob import TextBlob

from gensim.models import KeyedVectors # load the Stanford GloVe model


from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing import text, sequence

scaler = MinMaxScaler()


def _data_cleansing(df):
    import nltk
    nltk.download('stopwords')
    stop = set(stopwords.words('english'))
    punctuation = list(string.punctuation)
    stop.update(punctuation)

    def strip_html(text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    # Removing the square brackets
    def remove_between_square_brackets(text):
        return re.sub('\[[^]]*\]', '', text)

    # Removing punctuations
    def remove_punctuations(text):
        return re.sub('[^a-zA-z0-9\s]', '', text)

    # CONVERT TO LOWERCASE
    def lowerCase(text):
        text = text.lower()
        return text

    # Removing the stopwords from text
    def remove_stopwords(text):
        final_text = []
        for i in text.split():
            if i.strip().lower() not in stop:
                final_text.append(i.strip())
        return " ".join(final_text)

    # Removing the noisy text
    def denoise_text(text):
        text = strip_html(text)
        text = remove_between_square_brackets(text)
        text = remove_punctuations(text)
        text = lowerCase(text)
        # text = remove_stopwords(text)
        return text

    # remove top 10 most common words assuming they will not help in additional information

    # freq = pd.Series(' '.join(df['Description']).split()).value_counts()[:10]
    # freq = list(freq.index)
    # df['Description'] = df['Description'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
    #
    # # remove top 25 least common words assuming they will not help in additional information
    #
    # freq = pd.Series(' '.join(df['Description']).split()).value_counts()[-25:]
    # freq = list(freq.index)
    # df['Description'] = df['Description'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

    # Spell correct using Textblob
    df['Description'].apply(lambda x: str(TextBlob(x).correct()))

    # Lemmatize the sentences

    df['Description'] = df['Description'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    df['Description_denoised'] = df['Description'].apply(denoise_text)
    df.reset_index(inplace=True)
    return df

def desc2vector(desc, al, IS, gender, ET, CR, reloaded_word_vectors):
    mean_fvector = []
    mean_svector = []
    for nsent in range(len(desc)):
        # for isentence in desc[nsent].split():
        mean_svector = []
        input_list = desc[nsent].split()
        filtered_words = filter(lambda x: x in reloaded_word_vectors.vocab, input_list)
        sentence_len = 0.001
        vector = np.empty([200])
        for Sword in filtered_words:
            vector = np.add(vector, reloaded_word_vectors[Sword])
            sentence_len = sentence_len + 1
        mean_svector = np.divide(vector, sentence_len).tolist()
        mean_svector.extend([IS[nsent]])
        mean_svector.extend([gender[nsent]])
        mean_svector.extend([ET[nsent]])
        mean_svector.extend([CR[nsent]])
        mean_svector.extend([al[nsent]])

        mean_fvector.append(mean_svector)
    return mean_fvector

def predict_risk_class(recv_desc, recv_class_gender, recv_class_ind_sector, recv_class_emp_type, recv_class_cr, class_name):
    path = 'C:\\Users\\91992\\PycharmProjects\\CapstoneProject\\trained_models\\'
    max_features = 5500
    maxlen = 200
    temp_list = []
    temp_list.append(recv_desc)
    df1 = pd.DataFrame()
    df1 = pd.DataFrame(temp_list, columns=["Description"])
    df1 = _data_cleansing(df1)
    df1.reset_index(inplace=True)
    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(df1['Description_denoised'])
    tokenized_train = tokenizer.texts_to_sequences(df1['Description_denoised'])
    df_seq = sequence.pad_sequences(tokenized_train, maxlen=maxlen)

    reloaded_word_vectors = \
        KeyedVectors.load('C:\\Users\\91992\\PycharmProjects\\CapstoneProject\\training\\vectors.kv')
    outV = desc2vector(df1['Description_denoised'].tolist(), [0], [recv_class_ind_sector],[recv_class_gender], [recv_class_emp_type],
                         [recv_class_cr], reloaded_word_vectors)
    svm_feature_df = pd.DataFrame(outV)
    x_temp = svm_feature_df.iloc[:, :-1]
    _best_classifier = class_name

    if _best_classifier == "SVM":
        model_name = 'svm.model'
        loaded_model = pickle.load(open(path + model_name, 'rb'))
        predicted_class = loaded_model.predict(x_temp)[0]
        conf_level = round(loaded_model.predict_proba(x_temp).max(), 2)

    return predicted_class, conf_level


class ActionResetAllSlots(Action):
    def name(self):
        return "action_reset_all_slots"

    def run(self, dispatcher, tracker, domain):
        return [AllSlotsReset()]

class AccidentForm(FormAction):

    def name(self):
        return "accident_form"

    @staticmethod
    def required_slots(tracker):
        return ["date", "country", "ind_sector", "gender", "emp_type", "accident_type", "accident_level", "description"]

    def slot_mappings(self) -> Dict[Text, Union[Dict, List[Dict]]]:
        """A dictionary to map required slots to
            - an extracted entity
            - intent: value pairs
            - a whole message
            or a list of them, where a first match will be picked"""

        return {
            "date": [
                self.from_entity(entity="date"),
                self.from_intent(intent="deny", value="None"),
            ],
            "country": [
                self.from_text(not_intent="out_of_scope"),
            ],
            "ind_sector": [
                self.from_text(),
                self.from_intent(intent="deny", value="None"),
                # self.from_text(intent="deny"),
            ],
            "gender": [
                self.from_text(not_intent="out_of_scope"),
                # self.from_text(intent="deny"),
            ],
            "emp_type": [
                self.from_text(not_intent="out_of_scope"),
                # self.from_text(intent="deny"),
            ],
            "accident_type": [
                self.from_text(not_intent="out_of_scope"),
                # self.from_text(intent="deny"),
            ],
            "accident_level": [
                self.from_text(),
                self.from_intent(intent="deny", value="None"),
                # self.from_text(intent="deny"),
            ],
            "description": [
                self.from_text(not_intent="out_of_scope"),
                # self.from_text(intent="affirm"),
                # self.from_text(intent="deny"),
            ],
        }

    def submit(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any],
    ) -> List[Dict]:
        path = 'C:\\Users\\91992\\PycharmProjects\\CapstoneProject\\training\\'
        date = tracker.get_slot("date")
        country = tracker.get_slot("country")
        gender = tracker.get_slot("gender")
        emp_type = tracker.get_slot("emp_type")
        accident_type = tracker.get_slot("accident_type")
        ind_sector = tracker.get_slot("ind_sector")
        accident_level = tracker.get_slot("accident_level")
        description = tracker.get_slot("description")


        def insert_row(idx, df, df_insert):
            return df.iloc[:idx, ].append(df_insert).append(df.iloc[idx:, ]).reset_index(drop=True)

        labels = {0: 'I', 1: 'II', 2: 'III', 3: 'IV', 4: 'V', 5: 'VI'}

        classifier_name = "SVM"

        temp_dict_IS = {
            "Mining": 0,
            "Metals": 1,
            "Others": 2,
        }

        temp_dict_G = {
            "Male": 0,
            "Female": 1,
        }

        temp_dict_ET = {
            "Third Party": 1,
            "Employee": 0,
            "Third Party (Remote)": 2,
        }

        temp_dict_CR = {
            "Pressed (Heavy)": 0,
            "Manual Tools": 1,
            "Pressed (Medium)": 2,
            "Pressed (Minor)": 3,
            "Chemical substances": 4,
            "Vehicles and Mobile equipment": 5,
            "Cut": 6,
            "Projection": 7,
            "Fall prevention (same level)": 8,
            "Venomous Animals": 9,
            "Fall": 10,
            "Bees": 11,
            "Foot Twist": 12,
            "Rock fall (Heavy)": 13,
            "Pressurized Systems": 14,
            "Projection of fragments": 15,
            "remains of choco": 16,
            "Fall prevention": 17,
            "Projection of mud": 18,
            "Rock fall (Medium)": 19,
            "Rock fall (Minor)": 20,
            "Suspended Loads": 21,
            "Fall (Medium)": 22,
            "Burn": 23,
            "Machine Part Fall": 24,
            "Liquid Metal": 25,
            "Fall (High)": 26,
            "Pressurized Systems / Chemical Substances": 27,
            "Power lock": 28,
            "Cut (Minor)": 29,
            "Blocking and isolation of energy": 30,
            "Machine Protection": 31,
            "Electrical Shock": 32,
            "Projection/Manual Tools": 33,
            "Plates": 34,
            "\nNot applicable": 35,
            "Pressed (Low)": 36,
            "Poll": 37,
            "Projection/Choco": 38,
            "Burn (Minor)": 39,
            "Cooking": 40,
            "Projection/Burning": 41,
            "Pressed": 42,
            "Traffic": 43,
            "Electrical shock": 44,
            "Fall (Low)": 45,
            "Others": 46,
            "Confined space": 47,
            "Electrical installation": 48,
            "Individual protection equipment": 49
        }

        #setting default values
        class_gender = "Male"
        class_emp_type = "Employee"
        class_cr = "Pressed (Heavy)"
        class_ind_sector = "Mining"

        class_gender = temp_dict_G.get(gender,"0")
        class_emp_type = temp_dict_ET.get(emp_type,"0")
        class_cr = temp_dict_CR.get(accident_type,"0")
        class_ind_sector = temp_dict_IS.get(ind_sector,"0")

        pred_class, pred_conf_level = predict_risk_class(description, class_gender, class_ind_sector, class_emp_type, class_cr, classifier_name)

        print("Predicted Class:", labels[pred_class])
        print("Predicted confidence Level:", round(pred_conf_level * 100, 2))

        column_list = ['timestamp', 'Incident_no', 'Date of Accident', 'Country of Accident', 'Gender', 'Employee type', 'Critical Risk', 'Ind_sector', 'Severity',
                                      'Severity_Pred', 'Conf_level', 'description']

        accident_log_df = pd.read_csv(path + 'accident_log.csv', usecols=column_list)

        temp_timestamp = datetime.datetime.now()
        temp_inc_no    = "INC_" + str(uuid.uuid4().fields[-1])[:8]
        temp_date      =  date
        temp_country   =  country
        temp_vic_gender = gender
        temp_vic_emp_type = emp_type
        temp_vic_accident_type = accident_type
        temp_ind_sector = ind_sector
        temp_severity  =  accident_level
        temp_pred_sev  =  labels[pred_class]
        temp_conf_level = round(pred_conf_level * 100, 2)
        temp_desc       = description

        temp_df = pd.DataFrame(data=[[temp_timestamp, temp_inc_no, temp_date, temp_country, temp_vic_gender, temp_vic_emp_type, temp_vic_accident_type,
                                      temp_ind_sector, temp_severity, temp_pred_sev, temp_conf_level, temp_desc]], columns=column_list)


        new_accident_log_df = insert_row(accident_log_df.shape[0] + 1, accident_log_df, temp_df)


        new_accident_log_df.to_csv(path + 'accident_log.csv')

        msg = "Thanks, your answers have been recorded and Incident created! Pls note the incident no " + temp_inc_no

        # dispatcher.utter_message(msg)
        return [SlotSet("incno", temp_inc_no)]