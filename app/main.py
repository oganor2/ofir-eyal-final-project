import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
# import surprise
from predict import *

# st.write(os.getcwd())
### main page
image = Image.open('ofir-eyal-final-project-main/app/data/fig.png')
st.image(image, width=240)

title = '<p style="font-family:Impact; color:#8b2d2d; font-size: 44px;">Schedule Me</p>'
st.markdown(title, unsafe_allow_html=True)
user_id=(pd.read_csv('ofir-eyal-final-project-main/app/data/past_ranking.csv')['user_id']).max()+1

title2 = '<p style="font-family:Bahnschrift SemiBold; color:Black; font-size: 23px;">A recommendation system for your weekly class schedule</p>'
st.markdown(title2, unsafe_allow_html=True)
env = st.selectbox("select Default/Personalized", ['', 'Default', 'Personalized'])
if env == '':
    st.write('Default- a default schedule of courses based on the most popular courses among the student sample')
    st.write('Personalized- a schedule of courses based on your preferences')

    st.write(' - - - - - - - - - - - - - - - - - - - - - - - -  - - - - ')
    title3 = '<p style="font-family:Bahnschrift; color:Black; font-size: 18px;">With the help of our system, you can find the best courses for YOU- Taking into account your constraints, the system offers you courses that are most likely to be of interest to you based on ratings from a sample of students!</p>'
    st.markdown(title3, unsafe_allow_html=True)

if env == 'Default':
    if "submit_default" not in st.session_state:
        st.session_state["submit_default"] = False

    st.subheader('If you have limitations, please enter them')

    # more filters
    k = st.text_input('Please enter the required number of courses')
    if k == '':
        k = 5
    no_exam = st.checkbox('I want only courses without an exam')
    available_days = st.multiselect('Decide which days you want to study',
                                    ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
    days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    available_days = [days.index(d) + 1 for d in available_days]

    if st.button('submit'):
        st.session_state["submit_default"] = True
    change_part = False
    if st.session_state['submit_default'] == True and change_part == False:
        st.write("Here is a default courses schedule based on the most popular courses among the student sample")
        chosen = get_courses(user_id=user_id, k=int(k), no_exam=no_exam, days=available_days)

        # just make it looking good
        final_df = chosen[['course', 'course_id', 'exam', 'day']]
        days_dict = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday"}
        final_df['Day'] = final_df['day'].map(days_dict)
        exam_dict = {0: "No Exam", 1: "With Exam"}
        final_df['Exam'] = final_df['exam'].map(exam_dict)
        st.write(final_df[['course', 'course_id', 'Exam', 'Day']])

        new_dict = {}
        for day in days:
            new_dict[day] = []
        for i, row in final_df.iterrows():
            new_dict[row['Day']].append(row['course'])
        pivotted = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in new_dict.items()]))

        chosen_courses = list(chosen['course'])

        courses_dict = {'courses_list': chosen_courses}
        courses_list = pd.DataFrame.from_dict(courses_dict)

        change_part = st.checkbox('I have already took some of these courses')
        if change_part:
            unwanted_courses = st.multiselect('Please select the unwanted courses', chosen_courses)
            # st.write('new df without unwanted_courses')
            if st.button('get a new schdule'):

                chosen = get_courses(user_id=user_id, k=int(k), no_exam=no_exam, days=available_days,
                                     past_course=unwanted_courses)

                # just make it looking good
                final_df = chosen[['course', 'course_id', 'exam', 'day']]
                days_dict = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday"}
                final_df['Day'] = final_df['day'].map(days_dict)
                exam_dict = {0: "No Exam", 1: "With Exam"}
                final_df['Exam'] = final_df['exam'].map(exam_dict)

                # st.write('new df without the unwanted courses')
                st.write(final_df[['course', 'course_id', 'Exam', 'Day']])

                new_dict = {}
                for day in days:
                    new_dict[day] = []
                for i, row in final_df.iterrows():
                    new_dict[row['Day']].append(row['course'])
                pivotted = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in new_dict.items()]))

                chosen_courses = list(chosen['course'])

                courses_dict = {'courses_list': chosen_courses}
                courses_list = pd.DataFrame.from_dict(courses_dict)

        df_xlsx = to_excel(pivotted, final_df[['course', 'course_id', 'Exam', 'Day']])
        st.download_button(label='📥 Download courses schedule',
                           data=df_xlsx,
                           file_name='courses schedule.xlsx')

if env == 'Personalized':
    if "submit_person" not in st.session_state:
        st.session_state["submit_person"] = False

    st.subheader("Enter courses you have taken and rank how much did you like them:")


    @st.cache(allow_output_mutation=True)
    def get_data():
        return []


    # courses= df(course,course_id,day,test)
    # past_ranking= df(user_id, course_id,ranking)

    # user_id = past_ranking.user_id.max()+1
    courses=pd.read_csv('ofir-eyal-final-project-main/app/data/courses.csv')

    course = st.selectbox('Select a course', courses.course.unique())
    #course = st.text_input("course")
    rating = st.slider("rating", 0, 5)

    if st.button("Add row"):
        get_data().append({"user_id": user_id, "course": course, "ranking": rating})

    liked_courses = pd.DataFrame(get_data())
    if len(liked_courses) > 0:
        st.write(liked_courses[['course', 'ranking']])

    # more filters
    k = st.text_input('Please enter the required number of courses')
    if k == '':
        k = 5
    no_exam = st.checkbox('I want only courses without an exam')
    available_days = st.multiselect('Decide which days you want to study',
                                    ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
    days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    available_days = [days.index(d) + 1 for d in available_days]
    # st.write(user_id,k,no_exam,available_days)

    if st.button('submit'):
        st.session_state['submit_person'] = True

    change_part = False
    if st.session_state['submit_person'] == True and change_part == False:
        #st.write(liked_courses)
        chosen = get_courses(user_id=user_id, k=int(k), no_exam=no_exam, days=available_days, user_data=liked_courses)

        # just make it looking good
        final_df = chosen[['course', 'course_id', 'exam', 'day']]
        days_dict = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday"}
        final_df['Day'] = final_df['day'].map(days_dict)
        exam_dict = {0: "No Exam", 1: "With Exam"}
        final_df['Exam'] = final_df['exam'].map(exam_dict)
        st.write(final_df[['course', 'course_id', 'Exam', 'Day']])

        new_dict = {}
        for day in days:
            new_dict[day] = []
        for i, row in final_df.iterrows():
            new_dict[row['Day']].append(row['course'])
        pivotted = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in new_dict.items()]))

        chosen_courses = list(chosen['course'])

        courses_dict = {'courses_list': chosen_courses}
        courses_list = pd.DataFrame.from_dict(courses_dict)

        change_part = st.checkbox('I have already made some of these courses')
        if change_part:
            unwanted_courses = st.multiselect('Please select the unwanted courses', chosen_courses)
            st.write('new df without unwanted_courses')
            if st.button('get a new schdule'):
                chosen = get_courses(user_id=user_id, k=int(k), no_exam=no_exam, days=available_days,
                                     past_course=unwanted_courses, user_data=liked_courses)
                st.write('new df without the unwanted courses')

                # just make it looking good
                final_df = chosen[['course', 'course_id', 'exam', 'day']]
                days_dict = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday"}
                final_df['Day'] = final_df['day'].map(days_dict)
                exam_dict = {0: "No Exam", 1: "With Exam"}
                final_df['Exam'] = final_df['exam'].map(exam_dict)
                st.write(final_df[['course', 'course_id', 'Exam', 'Day']])

                new_dict = {}
                for day in days:
                    new_dict[day] = []
                for i, row in final_df.iterrows():
                    new_dict[row['Day']].append(row['course'])
                pivotted = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in new_dict.items()]))

                chosen_courses = list(chosen['course'])

                courses_dict = {'courses_list': chosen_courses}
                courses_list = pd.DataFrame.from_dict(courses_dict)

        df_xlsx = to_excel(pivotted, final_df[['course', 'course_id', 'Exam', 'Day']])
        st.download_button(label='📥 Download courses schedule',
                           data=df_xlsx,
                           file_name='courses schedule.xlsx')




