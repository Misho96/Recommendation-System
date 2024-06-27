from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from pyngrok import ngrok


app = Flask(__name__)

# Load your data and model artifacts
df = pd.read_csv("data/crunchbase-investments.csv", encoding='unicode_escape', low_memory=False)
df=df.drop(['company_permalink','investor_permalink',
        'investor_category_code',
        'funded_at', 'funded_month', 'funded_quarter',
        'funded_year','investor_state_code','investor_country_code',
        'investor_city','raised_amount_usd'],axis=1)

df=df.dropna(axis=0, subset=['company_category_code','company_state_code',
        'company_country_code','company_region',
        'company_city','funding_round_type','company_name','investor_name','investor_region'])



df = df.drop_duplicates(subset='company_name', keep='first')



df1=df

def clean_data(x):  # sourcery skip: assign-if-exp
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

features = ['company_category_code', 'company_country_code',
       'company_state_code', 'company_region', 'company_city', 'investor_name',
       'investor_region', 'funding_round_type']

for feature in features:
    df1[feature] = df1[feature].apply(clean_data)

df['metric'] = df[['company_category_code', 'company_country_code',
       'company_state_code', 'company_region', 'company_city', 'investor_name',
       'investor_region', 'funding_round_type']].apply(lambda x: " ".join(x), axis=1)

df2=df1.drop(['company_category_code', 'company_country_code',
       'company_state_code', 'company_region', 'company_city', 'investor_name',
       'investor_region', 'funding_round_type'],axis=1)
df2= df2.reset_index()
indices = pd.Series(df2.index, index=df2['company_name'])

cv = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
cv_matrix = cv.fit_transform(df['metric'])
cosine_sim = cosine_similarity(cv_matrix, cv_matrix)

@app.route('/recommend', methods=['GET'])
def recommend():
    company_name = request.args.get('company')
    if company_name not in indices:
        return jsonify({'error': 'Company not found'}), 404
    idx = indices[company_name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    company_indices = [i[0] for i in sim_scores]
    recommendations = df['company_name'].iloc[company_indices].tolist()
    return jsonify(recommendations)
investor_data = pd.read_csv("data/crunchbase-investments.csv", encoding='unicode_escape', low_memory=False)

investor_data=investor_data.drop(['company_permalink','investor_permalink',
        'company_country_code', 'company_state_code',
        'company_city','investor_category_code',
        'funded_at', 'funded_month', 'funded_quarter',
        'funded_year','raised_amount_usd'],axis=1)

investor_data=investor_data.dropna(axis=0, subset=['company_category_code','company_region',
                            'funding_round_type','company_name','investor_name','investor_region'])

investor_data = investor_data.drop_duplicates(subset='investor_name', keep='first')
investor_data1=investor_data
def clean_data(x):  # sourcery skip: assign-if-exp
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

investor_features =['company_name', 'company_category_code', 'company_region',
       'investor_country_code', 'investor_state_code',
       'investor_region', 'investor_city', 'funding_round_type']

for investor_features in investor_features:
    investor_data[investor_features] = investor_data[investor_features].apply(clean_data)

investor_data['metric'] =investor_data[['company_name', 'company_category_code', 'company_region',
        'investor_country_code', 'investor_state_code',
        'investor_region', 'investor_city', 'funding_round_type']].apply(lambda x: " ".join(x), axis=1)

investor_data2=investor_data1.drop(['company_name', 'company_category_code', 'company_region',
        'investor_country_code', 'investor_state_code',
        'investor_region', 'investor_city', 'funding_round_type'],axis=1)

cv_investor = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
cv_investor_matrix = cv_investor .fit_transform(investor_data2['metric'])
investor_cosine_sim = cosine_similarity(cv_investor_matrix, cv_investor_matrix)

investor_data2= investor_data.reset_index()
investors_indices = pd.Series(investor_data2.index, index=investor_data2['investor_name'])

@app.route('/recommend_investor', methods=['GET'])
def recommendation ():
    investor_name = request.args.get('investor')
    if investor_name not in investors_indices:
        return jsonify({'error': 'Investor not found'}), 404
    idx = investors_indices[investor_name]
    investor_sim_scores = list(enumerate(investor_cosine_sim[idx]))
    investor_sim_scores = sorted(investor_sim_scores, key=lambda x: x[1], reverse=True)
    investor_sim_scores = investor_sim_scores[1:11]
    investor_indices = [i[0] for i in investor_sim_scores]
    investors_recommendations = investor_data['investor_name'].iloc[investor_indices].tolist()
    return jsonify(investors_recommendations)

if __name__ == '__main__':
    NGROK_AUTH = "2iSH3VyvBssvA1m02RHZuHABviZ_7nKukBUTJEPQ4tHz86y8t"
    PORT = 5000
    ngrok.set_auth_token(NGROK_AUTH)
    tunnel = ngrok.connect(PORT , domain= "lucky-clear-anemone.ngrok-free.app")
    print("Public URL:",tunnel.public_url)
    app.run(host="0.0.0.0",port=5000)