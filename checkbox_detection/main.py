import streamlit as st
import cv2
import numpy as np
import pandas as pd

class CheckboxDetector:
    def __init__(self, image_content):
        self.image = cv2.imdecode(np.frombuffer(image_content, np.uint8), cv2.IMREAD_COLOR)
        self.thresh = self.threshold_image()

    def rescaleFrame(self):
        return cv2.resize(self.image, (600, 800), interpolation=cv2.INTER_LINEAR)

    def threshold_image(self):
        rescaled_img = self.rescaleFrame()
        gray = cv2.cvtColor(rescaled_img, cv2.COLOR_BGR2GRAY)
        rest, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        return thresh

    def process_checkdata(self, y_start, y_end, rows):
        data = np.zeros((rows, 31), dtype=int)

        for i, y in enumerate(np.linspace(y_start, y_end, rows)):
            for j, x in enumerate(np.linspace(102, 528, 31)):
                x_int = int(x)
                y_int = int(y)
                roi = self.thresh[y_int:y_int + 10, x_int:x_int + 10]
                black_pixels = np.sum(roi == 0)
                checkbox_status = 1 if black_pixels > 0 else 0
                data[i, j] = checkbox_status
        return data

    def screen_time_df(self):

        screen_time = self.process_checkdata(108, 170, 4)
        screen_time_df = pd.DataFrame(screen_time)
        new_index = ['2 to 3 hours', '3 to 5 hours', '5 to 8 hours', 'More than 8 hours']
        screen_time_df.index = new_index

        for i in screen_time_df.columns:
            screen_time_df.rename(columns={i: f'Day {i + 1}'}, inplace=True)
        screen_time_df.rename_axis("Screen Time",inplace=True)
        return screen_time_df

    def haleness_df(self):
        haleness = self.process_checkdata(210, 253, 3)
        haleness_df = pd.DataFrame(haleness)
        new_index = ['Eye break rule (20-20-20)', 'Exposure to Sunlight', 'Enable Dark Mode at Night']
        haleness_df.index = new_index

        for i in haleness_df.columns:
            haleness_df.rename(columns={i: f'Day {i + 1}'}, inplace=True)
        haleness_df.rename_axis("Haleness",inplace=True)
        return haleness_df

    def water_consumption_df(self):
        water_consumption = self.process_checkdata(294, 334, 3)
        water_consumption_df = pd.DataFrame(water_consumption)
        new_index = ['<1 litres', '<2 litres', '<5 litres']
        water_consumption_df.index = new_index

        for i in water_consumption_df.columns:
            water_consumption_df.rename(columns={i: f'Day {i + 1}'}, inplace=True)
        water_consumption_df.rename_axis("Water Consumption", inplace=True)
        return water_consumption_df

    def physical_and_mental_hassle_df(self):
        physical_and_mental_hassle = self.process_checkdata(376, 457, 5)
        physical_and_mental_hassle_df = pd.DataFrame(physical_and_mental_hassle)
        new_index = ['Eye Strain', 'Back Pain', 'Stressed Out', 'Battery Anxiety', 'Thumb and Wrist Discomfort']
        physical_and_mental_hassle_df.index = new_index

        for i in physical_and_mental_hassle_df.columns:
            physical_and_mental_hassle_df.rename(columns={i: f'Day {i + 1}'}, inplace=True)
        physical_and_mental_hassle_df.rename_axis("Physical and Mental Hassle",inplace=True)
        return physical_and_mental_hassle_df

    def screen_free_meals_df(self):
        screen_free_meals = self.process_checkdata(498, 536, 3)
        screen_free_meals_df = pd.DataFrame(screen_free_meals)
        new_index = ['Breakfast', 'Lunch', 'Dinner']
        screen_free_meals_df.index = new_index

        for i in screen_free_meals_df.columns:
            screen_free_meals_df.rename(columns={i: f'Day {i + 1}'}, inplace=True)
        screen_free_meals_df.rename_axis("Screen Free Meals",inplace=True)
        return screen_free_meals_df

    def meal_time_journal_df(self):
        meal_time_journal = self.process_checkdata(596, 646, 6)
        meal_time_journal_df = pd.DataFrame(meal_time_journal)
        new_index = ['6 to 8 am', '6 to 8 pm', '8 to 10 am', '8 to 10 pm', '10 to 11 am', '10 to 11 pm']
        meal_time_journal_df.index = new_index

        for i in meal_time_journal_df.columns:
            meal_time_journal_df.rename(columns={i: f'Day {i + 1}'}, inplace=True)
        meal_time_journal_df.rename_axis("Meal Time Journal",inplace=True)
        return meal_time_journal_df

    def sleep_quality_indicators_df(self):
        sleep_quality_indicators = self.process_checkdata(684, 724, 3)
        sleep_quality_indicators_df = pd.DataFrame(sleep_quality_indicators)
        new_index = ['Deep Sleep (>7 Hours)', 'Light Sleep (<5 Hours)', 'Often Awake (<3 Hours)']
        sleep_quality_indicators_df.index = new_index

        for i in sleep_quality_indicators_df.columns:
            sleep_quality_indicators_df.rename(columns={i: f'Day {i + 1}'}, inplace=True)
        sleep_quality_indicators_df.rename_axis("Sleep Quality Indicators",inplace=True)
        return sleep_quality_indicators_df

    def social_interaction_df(self):
        social_interaction = self.process_checkdata(766, 786, 2)
        social_interaction_df = pd.DataFrame(social_interaction)
        new_index = ['More on social media', 'More in-person conversation']
        social_interaction_df.index = new_index

        for i in social_interaction_df.columns:
            social_interaction_df.rename(columns={i: f'Day {i + 1}'}, inplace=True)
        social_interaction_df.rename_axis("Social Interaction",inplace=True)
        return social_interaction_df

    def combined_data_df(self):
        output = {'Screen Time': self.screen_time_df(),
                  'Haleness': self.haleness_df(),
                  'Water Consumption': self.water_consumption_df(),
                  'Physical and Mental Hassle': self.physical_and_mental_hassle_df(),
                  'Screen Free Meals': self.screen_free_meals_df(),
                  'Meal Time Journal': self.meal_time_journal_df(),
                  'Sleep Quality Indicators': self.sleep_quality_indicators_df(),
                  'Social Interaction': self.social_interaction_df()
                  }

        combined_data = pd.concat(output)
        combined_data.rename_axis(["Category","Subcategory"],inplace=True)
        return combined_data

def main():
    st.header("🧘 Digital Wellbeing Tracker")
    with st.sidebar:
        st.title("Digital Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])


    if uploaded_file is not None:
        checkbox_detector = CheckboxDetector(uploaded_file.getvalue())
        df_options = [
            "Screen Time",
            "Haleness",
            "Water Consumption",
            "Physical and Mental Hassle",
            "Screen Free Meals",
            "Meal Time Journal",
            "Sleep Quality Indicators",
            "Social Interaction",
            "Combined DataFrame"
        ]
        selected_df = st.selectbox("Select a dataframe:", df_options, index=None, placeholder="Select your tracker")

        if selected_df == "Screen Time":
            st.write(checkbox_detector.screen_time_df())
        elif selected_df == "Haleness":
            st.write(checkbox_detector.haleness_df())
        elif selected_df == "Water Consumption":
            st.write(checkbox_detector.water_consumption_df())
        elif selected_df == "Physical and Mental Hassle":
            st.write(checkbox_detector.physical_and_mental_hassle_df())
        elif selected_df == "Screen Free Meals":
            st.write(checkbox_detector.screen_free_meals_df())
        elif selected_df == "Meal Time Journal":
            st.write(checkbox_detector.meal_time_journal_df())
        elif selected_df == "Sleep Quality Indicators":
            st.write(checkbox_detector.sleep_quality_indicators_df())
        elif selected_df == "Social Interaction":
            st.write(checkbox_detector.social_interaction_df())
        elif selected_df == "Combined DataFrame":
            st.write(checkbox_detector.combined_data_df())


if __name__ == "__main__":
    main()
