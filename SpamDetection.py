import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

########### DATASET
data = pd.read_csv("spam.csv")
##----------------------------------------
# print (data.head())
# print(data.shape)
##------------------------------------------

data.drop_duplicates(inplace=True) # remove the duplicates 

###inplace = True it is like do the changements in the same dataset 
## we can do data1 = data.drop_duplicates(inplace=True)
# print(data.shape)
#print(data.isnull().sum()) #to check is there any null records
data['Category'] = data['Category'].replace(['ham','spam'],['Not Spam','Spam']) # change the names 
#print (data.head())

mess = data['Message'] #input dataset
cat = data ['Category'] #output dataset

##-------Split the dataset 80% train dataset and 20% test dataset
(mess_train,mess_test,cat_train,cat_test) = train_test_split(mess, cat, test_size=0.2)
##---------convert test to numerical data using countvectorizer
cv =CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)

########### CREATE THE MODEL 
model = MultinomialNB()
##--------------Train the model 
model.fit(features , cat_train)   # feature is input dataset , cat_train is output dataset

###########   TEST THE MODEL 
#after the model is ready 
features_test = cv.transform(mess_test) # transform the test dataset into vector
#print(model.score(features_test , cat_test)) # pass the test data into the model and validate it against the actual test data and print the score of the model based on the accurancy


###########PREDICT DATA ON REAL TIME 


#message = cv.transform(['Congratulations , you won a lottery ']).toarray()
#result = model.predict(message)
#print(result)
def predict(message):
    input_message = cv.transform([message]).toarray()
    result = model.predict(input_message)
    return result 



#webpage 
## here we will use streamlit to create a simple webpage to test the model
# =========================
# WEBPAGE
# =========================

st.set_page_config(
    page_title="Spam Detector",
    page_icon="📧"
)

st.title("📧 Spam Message Detector")

st.write("Enter a message below and check whether it is spam or not.")

st.divider()

input_message = st.text_area(
    "Message",
    placeholder="Write your message here...",
    height=150
)

if st.button("Check Message"):

    if input_message.strip() == "":
        st.warning("Please enter a message.")

    else:
        output = predict(input_message)

        if output[0] == "Spam":
            st.error("🚨 This message is SPAM.")

        else:
            st.success("✅ This message is NOT SPAM.")

st.divider()

st.caption("Powered by Machine Learning • Multinomial Naive Bayes")
