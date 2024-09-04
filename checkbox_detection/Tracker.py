import streamlit as st
import cv2
import numpy as np
from io import BytesIO


class CheckboxDetector:
    def __init__(self, image_content):
        self.image = cv2.imdecode(np.frombuffer(image_content, np.uint8), cv2.IMREAD_COLOR)
        self.thresh = self.threshold_image()

    def rescaleFrame(self):
        return cv2.resize(self.img, (600, 800), interpolation=cv2.INTER_LINEAR)

    def threshold_image(self):
        rescaled_img = self.rescaleFrame()
        gray = cv2.cvtColor(rescaled_img, cv2.COLOR_BGR2GRAY)
        rest, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        return thresh

    def screen_time(self):
        screen_time_data = np.zeros((4, 31), dtype=int)

        for i, y in enumerate(np.linspace(108, 170, 4)):
            for j, x in enumerate(np.linspace(102, 528, 31)):
                x_int = int(x)
                y_int = int(y)
                roi = self.thresh[y_int:y_int + 10, x_int:x_int + 10]
                black_pixels = np.sum(roi == 0)
                checkbox_status = 1 if black_pixels > 0 else 0
                screen_time_data[i, j] = checkbox_status
        return screen_time_data

    def show_screen_time_data(self):
        screen = self.screen_time()
        print(screen)
        st.write("Screen Time Data:")
        st.write(screen)
        st.write("\n")

    def haleness(self):
        haleness_data = np.zeros((3, 31), dtype=int)

        for i, y in enumerate(np.linspace(210, 253, 3)):
            for j, x in enumerate(np.linspace(102, 528, 31)):
                x_int = int(x)
                y_int = int(y)
                roi = self.thresh[y_int:y_int + 10, x_int:x_int + 10]
                black_pixels = np.sum(roi == 0)
                checkbox_status = 1 if black_pixels > 0 else 0
                haleness_data[i, j] = checkbox_status
        return haleness_data

    def show_haleness_data(self):
        halen = self.haleness()
        print(halen)
        st.write("Haleness Data:")
        st.write(halen)
        st.write("\n")

    def water_consumption(self):
        water_consumption_data = np.zeros((3, 31), dtype=int)

        for i, y in enumerate(np.linspace(294, 334, 3)):
            for j, x in enumerate(np.linspace(102, 528, 31)):
                x_int = int(x)
                y_int = int(y)
                roi = self.thresh[y_int:y_int + 10, x_int:x_int + 10]
                black_pixels = np.sum(roi == 0)
                checkbox_status = 1 if black_pixels > 0 else 0
                water_consumption_data[i, j] = checkbox_status
        return water_consumption_data

    def show_water_consumption_data(self):
        water = self.water_consumption()
        print(water)
        st.write("Water Consumption Data:")
        st.write(water)
        st.write("\n")

    def physical_and_mental_hassle(self):
        physical_and_mental_hassle_data = np.zeros((5, 31), dtype=int)

        for i, y in enumerate(np.linspace(376, 457, 5)):
            for j, x in enumerate(np.linspace(102, 528, 31)):
                x_int = int(x)
                y_int = int(y)
                roi = self.thresh[y_int:y_int + 10, x_int:x_int + 10]
                black_pixels = np.sum(roi == 0)
                checkbox_status = 1 if black_pixels > 0 else 0
                physical_and_mental_hassle_data[i, j] = checkbox_status
        return physical_and_mental_hassle_data

    def show_physical_and_mental_hassle_data(self):
        physical = self.physical_and_mental_hassle()
        print(physical)
        st.write("Physical and Mental Hassle Data:")
        st.write(physical)
        st.write("\n")

    def screen_free_meals(self):
        screen_free_meals_data = np.zeros((3, 31), dtype=int)

        for i, y in enumerate(np.linspace(498, 536, 3)):
            for j, x in enumerate(np.linspace(102, 528, 31)):
                x_int = int(x)
                y_int = int(y)
                roi = self.thresh[y_int:y_int + 10, x_int:x_int + 10]
                black_pixels = np.sum(roi == 0)
                checkbox_status = 1 if black_pixels > 0 else 0
                screen_free_meals_data[i, j] = checkbox_status
        return screen_free_meals_data

    def show_screen_free_meal_data(self):
        screen_free = self.screen_free_meals()
        print(screen_free)
        st.write("Screen Free Meal Data:")
        st.write(screen_free)
        st.write("\n")

    def meal_time_journal(self):
        meal_time_journal_data = np.zeros((6, 31), dtype=int)

        for i, y in enumerate(np.linspace(596, 646, 6)):
            for j, x in enumerate(np.linspace(102, 528, 31)):
                x_int = int(x)
                y_int = int(y)
                roi = self.thresh[y_int:y_int + 10, x_int:x_int + 10]
                black_pixels = np.sum(roi == 0)
                checkbox_status = 1 if black_pixels > 0 else 0
                meal_time_journal_data[i, j] = checkbox_status
        return meal_time_journal_data

    def show_meal_time_journal_data(self):
        meal = self.meal_time_journal()
        print(meal)
        st.write("Meal Time Journal Data:")
        st.write(meal)
        st.write("\n")

    def sleep_quality_indicators(self):
        sleep_quality_indicators_data = np.zeros((3, 31), dtype=int)

        for i, y in enumerate(np.linspace(684, 724, 3)):
            for j, x in enumerate(np.linspace(102, 528, 31)):
                x_int = int(x)
                y_int = int(y)
                roi = self.thresh[y_int:y_int + 10, x_int:x_int + 10]
                black_pixels = np.sum(roi == 0)
                checkbox_status = 1 if black_pixels > 0 else 0
                sleep_quality_indicators_data[i, j] = checkbox_status
        return sleep_quality_indicators_data

    def show_sleep_quality_indicators_data(self):
        sleep = self.sleep_quality_indicators()
        print(sleep)
        st.write("Sleep Quality Indicators Data:")
        st.write(sleep)
        st.write("\n")

    def social_interaction(self):
        social_interaction_data = np.zeros((2, 31), dtype=int)

        for i, y in enumerate(np.linspace(766, 786, 2)):
            for j, x in enumerate(np.linspace(102, 528, 31)):
                x_int = int(x)
                y_int = int(y)
                roi = self.thresh[y_int:y_int + 10, x_int:x_int + 10]
                black_pixels = np.sum(roi == 0)
                checkbox_status = 1 if black_pixels > 0 else 0
                social_interaction_data[i, j] = checkbox_status
        return social_interaction_data

    def show_social_interaction_data(self):
        social = self.social_interaction()
        print(social)
        st.write("Social Interaction Data:")
        st.write(social)
        st.write("\n")

    # img_path = "track.png"
    # img = cv2.imread(img_path)
    # detector=CheckboxDetector(img)


def main():
    st.title("Digital Wellbeing Tracker")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    show_button = st.button("Tracing Data")
    if uploaded_file is not None:
        if show_button:
            detector = CheckboxDetector(uploaded_file.read())

            selected_option = st.selectbox(
                "Select an option:",
                ["Screen Time", "Haleness", "Water Consumption", "Physical and Mental Hassle", "Screen Free Meals",
                 "Meal Time Journal", "Sleep Quality Indicators", "Social Interaction"]
            )


            if selected_option == "Screen Time":
                detector.show_screen_time_data()
            elif selected_option == "Haleness":
                detector.show_haleness_data()
            elif selected_option == "Water Consumption":
                detector.show_water_consumption_data()
            elif selected_option == "Physical and Mental Hassle":
                detector.show_physical_and_mental_hassle_data()
            elif selected_option == "Screen Free Meals":
                detector.show_screen_free_meal_data()
            elif selected_option == "Meal Time Journal":
                detector.show_meal_time_journal_data()
            elif selected_option == "Sleep Quality Indicators":
                detector.show_sleep_quality_indicators_data()
            elif selected_option == "Social Interaction":
                detector.show_social_interaction_data()


if __name__ == "__main__":
    main()
