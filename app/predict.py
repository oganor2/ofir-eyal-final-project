import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader, SVDpp, KNNBaseline, SlopeOne
from math import isnan
from surprise import Dataset, NormalPredictor, Reader
from surprise.model_selection import cross_validate

def get_courses(user_id, k=5, user_data=None,
                no_exam=False, days='', past_course=None):
    r = Reader(sep=",", skip_lines=1)
    X = Dataset.load_from_df(pd.read_csv('data/past_ranking.csv'), reader=r).build_full_trainset()
    algo = SVD()
    algo.fit(X)
    courses = pd.read_csv('data/courses.csv')


    if no_exam:
        courses = courses[courses['exam'] == 0]
    if len(days) > 0:
        courses = courses[courses['day'].isin(days)]
    if past_course is not None:
        courses = courses[~courses['course'].isin(past_course)]
    if user_data is not None:
        data = pd.merge(user_data, courses, on='course')[['user_id', 'course_id', 'ranking']]
        courses = courses[~courses['course'].isin(list(user_data['course']))]
        X_user = Dataset.load_from_df(data, reader=r).build_full_trainset()
        algo.fit(X_user)
    est = {}
    for i, row in courses.iterrows():
        est[row['course_id']] = algo.predict(user_id, row['course_id']).est
        est[row['course_id']] = 2

    courses['rating'] = est.values()
    courses = courses.sort_values(by=['rating', 'course_id'])

    return courses.iloc[:k]

def to_excel(df, df2):
    import xlsxwriter
    from io import BytesIO
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='final_schedule')
    df2.to_excel(writer, index=False, sheet_name='courses_data')

    workbook = writer.book
    worksheet = writer.sheets['final_schedule']
    format1 = workbook.add_format({'num_format': '0.00'})
    worksheet.set_column('A:A', None, format1)
    writer.save()
    processed_data = output.getvalue()
    return processed_data

