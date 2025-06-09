# gmail_cleaner.py

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
import pickle


# Gmaail auth
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

def gmail_authenticate():
    from google_auth_oauthlib.flow import InstalledAppFlow
    import webbrowser

    try:
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        auth_url, _ = flow.authorization_url(prompt='consent')
        print(f"🔗 Please visit this URL to authorize the app:\n{auth_url}\n")

        webbrowser.open(auth_url) 

        code = input("📥 Paste the authorization code here: ")
        flow.fetch_token(code=code)

        creds = flow.credentials
        service = build('gmail', 'v1', credentials=creds)
        return service
    except Exception as e:
        print(f"❌ Error during authentication: {e}")
        raise



def get_emails(service, label_ids=['INBOX'], max_results=50):
    messages = service.users().messages().list(
        userId='me', labelIds=label_ids, maxResults=max_results
    ).execute().get('messages', [])

    emails = []
    for msg in messages:
        msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
        payload = msg_data.get('payload', {})
        headers = payload.get('headers', [])
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), "")
        snippet = msg_data.get('snippet', "")
        combined_text = subject + " " + snippet
        emails.append((msg['id'], combined_text))
    return emails


def delete_promotions(service, model, vectorizer, emails):
    for msg_id, text in emails:
        features = vectorizer.transform([text])
        prediction = model.predict(features)[0]
        if prediction == 1:
            service.users().messages().trash(userId='me', id=msg_id).execute()
            print(f"🗑️ Moved to Trash: {text[:60]}...")
        else:
            print(f"✅ ❤Kept: {text[:60]}...")


if __name__ == "__main__":
  
    service = gmail_authenticate()

   
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

   
    emails = get_emails(service, label_ids=['INBOX'], max_results=50)

   
    delete_promotions(service, model, vectorizer, emails)
