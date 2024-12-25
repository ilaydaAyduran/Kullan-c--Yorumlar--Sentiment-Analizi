import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# CSV dosyasını okuma
df = pd.read_csv("csv dosyanızın yolunu girmelisiniz", sep=";", on_bad_lines='skip')

# Sütun adlarını temizleme
df.columns = df.columns.str.strip()

# Sentiment etiketleri oluşturma
df['Sentiment'] = df['Answer1Value'].apply(lambda x: 'pozitif' if x >= 3 else 'negatif')

# Yalnızca yorum ve sentiment sütunlarını alıyoruz
df = df[['AnswerText', 'Sentiment']]

# Boş verileri kontrol etme ve temizleme
df.dropna(subset=['AnswerText'], inplace=True)

# Yorumları ve etiketleri ayırma
X = df['AnswerText']
y = df['Sentiment']

# Yorumları sayısal verilere dönüştürme (Bag of Words)
vectorizer = CountVectorizer()
X_vect = vectorizer.fit_transform(X)

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Naive Bayes modelini eğitme
model = MultinomialNB()
model.fit(X_train, y_train)

# Test verileri ile tahmin yapma
y_pred = model.predict(X_test)

# Sonuçları değerlendirme
print(f"Doğruluk Oranı: {accuracy_score(y_test, y_pred)}")
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

# Kullanıcıdan yorum al ve sınıflandır
def classify_comment(comment):
    # Yorumları sayısal verilere dönüştürme
    comment_vect = vectorizer.transform([comment])
    
    # Model ile tahmin yapma
    prediction = model.predict(comment_vect)
    
    return prediction[0]

# Kullanıcıdan yorum al
user_comment = input("Yorumunuzu girin: ")

# Yorum sınıflandırmasını yapın
result = classify_comment(user_comment)
print(f"Yorumunuzun analizi: {result}")
