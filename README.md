**Customer Satisfaction Prediction**

This repository contains the code and documentation for the Customer Satisfaction Prediction project, which aims to predict customer satisfaction ratings using machine learning models based on historical customer support ticket data.

**Project Overview:**

          Customer satisfaction is crucial for businesses to improve their support services and retain customers. In this project, we analyze customer support tickets and build machine learning models to predict satisfaction ratings. This predictive model can help businesses proactively address potential issues and improve overall customer experience.The project was completed through the Unified Mentor platform and follows a structured approach for end-to-end machine learning model development.

**Project Workflow**

**Tools and Technologies**

**Setup Instructions**

**Results and Insights**

**Dataset Description:**
              
              The dataset includes customer support tickets for various tech products. It provides information about:

                    Customer details (e.g., age, gender, email)
                    Product purchased
                    Ticket type, status, priority, and channel
                    Resolution details and response times
                    Customer satisfaction ratings (on a scale of 1 to 5)
                    
**Features:**
                    Ticket ID: Unique identifier for each ticket
                    Customer Name: Name of the customer
                    Customer Age: Age of the customer
                    Product Purchased: Product associated with the ticket
                    Ticket Type: Type of the ticket (e.g., technical issue, billing inquiry)
                    Ticket Status: Current status of the ticket
                    Time to Resolution: Time taken to resolve the ticket
                    Customer Satisfaction Rating: Customer-provided rating after ticket resolution
**Project Workflow:**

                    Data Collection and Preprocessing
                    Loaded the dataset and handled missing values
                    Converted date columns to appropriate formats
                    Encoded categorical variables
                    Exploratory Data Analysis (EDA)
                    Visualized trends in ticket types, resolution times, and satisfaction ratings
                    Analyzed correlations between features
                    
**Feature Engineering:**

                    Created new features such as ticket duration and response time categories
                    Selected key features for model building
                    Model Building

**Trained and evaluated multiple machine learning models:**

                    Linear Regression
                    Random Forest Regressor
                    Gradient Boosting Regressor
                    Used cross-validation to tune hyperparameters
                    
**Model Evaluation:**

                    Evaluated models using metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score
                    Selected the best model based on performance
                    
**Tools and Technologies:**
                    Languages: Python, SQL
                    Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
                    Tools: VS Code, Jupyter Notebook, Excel
                    
Setup Instructions
Clone the repository:
bash
Copy code
git clone https://github.com/yourusername/customer-satisfaction-prediction.git
Navigate to the project directory:
bash
#Copy code
cd customer-satisfaction-prediction
Install required dependencies:
bash
Copy code
pip install -r requirements.txt
Run the Jupyter Notebook:
bash
Copy code
jupyter notebook
Results and Insights
The Random Forest Regressor model provided the best performance with an R² score of 0.87 and RMSE of 0.35.
Key factors influencing customer satisfaction include ticket resolution time, product type, and ticket priority.
Visualizations highlighted that faster response and resolution times correlate with higher satisfaction ratings.
Future Enhancements
Implement advanced NLP techniques to analyze ticket descriptions for better feature extraction.
Build a real-time prediction system that businesses can integrate into their support platforms.
Explore deep learning models for improved accuracy.
Contact
If you have any questions or suggestions, feel free to reach out!

Name: HANA M

Email: hanarbeek2603@gmail.com

LinkedIn: www.linkedin.com/in/hana-mohamed-she-her-509a06249    

