import streamlit as st
from Home import face_rec
import pandas as pd

# st.set_page_config(page_title='Reporting',layout='wide')
st.subheader('Reporting')

    # retrive logs data and show in report.py
    # extract data from redis list
name = 'attendance:logs'
def load_logs(name,end=-1):
    logs_list = face_rec.r.lrange(name,start=0,end=end) # extract all data from the redis database
    return logs_list

    # tabs to show the info
tab1, tab2, tab3 = st.tabs(['Registered Data','Logs', 'Attendance Report'])

with tab1:
    if st.button('Refresh Data'):
        # retrive data from redis db
        with st.spinner('Retriving Data from Redis DB ...'):
            redis_face_db = face_rec.retrive_data(name='academy:register')
            st.dataframe(redis_face_db[['Name','Role']])

with tab2:  
    if st.button('Refresh Logs'):
        st.write(load_logs(name=name))


with tab3:
    st.subheader('Attendance Report')
        
     # load logs into attributes logs_list
    logs_list = load_logs(name=name)
        
    # 1 convert the logs that in list of bytes into list of string
    convert_bytes_to_string = lambda x: x.decode('utf-8')
    logs_list_string = list(map(convert_bytes_to_string, logs_list))
        
    # 2 split string by @ and create nested list
    split_string = lambda x: x.split('@')
    logs_nested_list = list(map(split_string, logs_list_string))
        
     # convert nested list info into dataframe
    logs_df = pd.DataFrame(logs_nested_list, columns= ['Name','Role','Timestamp'])
        
    # 3 time based analysis report
    logs_df['Timestamp'] = pd.to_datetime(logs_df['Timestamp'])
    logs_df['Date'] = logs_df['Timestamp'].dt.date
        
    # 3.1 Calculate intime and outtime
    # Intime: at which person is first detected in that day (min timestamp of the day)
    # Out Time at which person is last detected in that day (max timestamp of the day)
        
    report_df = logs_df.groupby(by=['Date','Name','Role']).agg(
        In_time = pd.NamedAgg('Timestamp','min'), # in time
        Out_time = pd.NamedAgg('Timestamp','max') # out time
    ).reset_index()
        
    report_df['In_time'] = pd.to_datetime(report_df['In_time'])
    report_df['Out_time'] = pd.to_datetime(report_df['Out_time'])
        
    report_df['Duration'] = report_df['Out_time'] - report_df['In_time']
        
    # 4 Marking person is Present or absent
    all_dates = report_df['Date'].unique()
    name_role = report_df[['Name','Role']].drop_duplicates().values.tolist()
        
    date_name_role_zip = []
    for dt in all_dates:
        for name, role in name_role:
            date_name_role_zip.append([dt, name, role])
                
    date_name_role_zip_df = pd.DataFrame(date_name_role_zip, columns=['Date','Name','Role'])
        
    #left join with report_df
        
    date_name_role_zip_df = pd.merge(date_name_role_zip_df, report_df, how='left',on=['Date','Name','Role'])
        
        
    # duration
    # hours
    date_name_role_zip_df['Duration_seconds'] = date_name_role_zip_df['Duration'].dt.seconds
    date_name_role_zip_df['Duration_hours'] = date_name_role_zip_df['Duration_seconds'] / (60*60)
        
    def status_marker(x):
        if pd.Series(x).isnull().all():
            return 'Absent'
            
        elif x >= 0 and x < 1:
            return 'Absent'
            
        elif x >= 1:
            return 'Present'
            
    date_name_role_zip_df['Status'] = date_name_role_zip_df['Duration_hours'].apply(status_marker)
        
    st.dataframe(date_name_role_zip_df)

        
