from __future__ import print_function

import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import googleapiclient.discovery
import re


# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']


# The ID of a sample document.
SAMPLE_SPREADSHEET_ID = '1zkR7qce0SHbHOrjmt3aNgc4l_Z_hK0Aza_q1bY0pZPQ' # people data


def get_people_responses(max_results=10):
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('sheets', 'v4', credentials=creds)

        # Call the Sheets API
        
        sheet = service.spreadsheets()
        result = sheet.values().get(spreadsheetId=SAMPLE_SPREADSHEET_ID,
                                    range=f"sheet1!B2:F{max_results}").execute() # B:General Problem  	B:Solution	   C:Name
        values = result.get('values', [])
        
        responses = []
        for row in values:
            if len(row) < 3:
                continue
            dict = {}
            dict['problem'] = row[0]
            # dict['challenge'] = row[1]
            # dict['chall_prob'] = row[2]
            dict['solution'] = row[1]
            dict['email'] = row[2]
            responses.append(dict)
        
        # print(responses)

        if not values:
            print('No data found.')
            return
        return responses
        for row in values:
            print(row)
            
    except HttpError as err:
        print(err)

def main():
    problems = get_people_responses()
    print(problems)

if __name__ == '__main__':
    main()