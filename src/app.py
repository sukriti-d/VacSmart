import streamlit as st
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

def plot_confidence_gauge(prediction, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction * 100,
        title={'text': title},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#2E7D32"},
            'steps': [
                {'range': [0, 33], 'color': "#FF4B4B"},
                {'range': [33, 66], 'color': "#FFA500"},
                {'range': [66, 100], 'color': "#2E7D32"}
            ]
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
        font={'size': 16}
    )
    
    return fig

# Page configuration
st.set_page_config(
    page_title="VacSmart",
    #page_icon="💉",
    layout="wide"
)

# Custom CSS 
st.markdown("""
<style>
.stButton button {
    background-color: #4CAF50;
    color: white;
    border-radius: 20px;
    padding: 0.5rem 2rem;
    border: none;
    transition: all 0.3s ease;
}
.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}
.vaccine-info {
    background-color: #F1F3F4;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
}
div.stMarkdown {
    padding: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_data():
    with open('models/h1n1_model.pkl', 'rb') as f:
        h1n1_model = pickle.load(f)
    with open('models/seasonal_model.pkl', 'rb') as f:
        seasonal_model = pickle.load(f)
    train_features = pd.read_csv('data/training_set_features.csv')
    return h1n1_model, seasonal_model, train_features

h1n1_model, seasonal_model, train_features = load_data()

# Main app header
st.title(' VacSmart')
st.markdown("### Made by Manzaar")


# Tabs for different sections
tab1, tab2, tab3 = st.tabs(["Prediction Tool", "Vaccine Information", "FAQ Assistant"])

with tab1:
    # User input collection
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Demographics")
        age_group = st.selectbox('Age Group', train_features['age_group'].unique())
        education = st.selectbox('Education Level', train_features['education'].unique())
        race = st.selectbox('Race/Ethnicity', train_features['race'].unique())
        sex = st.selectbox('Gender', train_features['sex'].unique())

    with col2:
        st.subheader("🏥 Health Information")
        h1n1_concern = st.slider('H1N1 Concern Level (0-3)', 0, 3, 1)
        h1n1_knowledge = st.slider('H1N1 Knowledge Level (0-3)', 0, 3, 1)
        doctor_recc_h1n1 = st.checkbox('Doctor Recommended H1N1 Vaccine')
        doctor_recc_seasonal = st.checkbox('Doctor Recommended Seasonal Vaccine')
        chronic_med_condition = st.checkbox('Has Chronic Medical Condition')
        health_worker = st.checkbox('Healthcare Worker')
        health_insurance = st.checkbox('Has Health Insurance')


    # Create input dataframe
    data = {
        'h1n1_concern': h1n1_concern,
        'h1n1_knowledge': h1n1_knowledge,
        'behavioral_antiviral_meds': 0,
        'behavioral_avoidance': 0,
        'behavioral_face_mask': 0,
        'behavioral_wash_hands': 0,
        'behavioral_large_gatherings': 0,
        'behavioral_outside_home': 0,
        'behavioral_touch_face': 0,
        'doctor_recc_h1n1': int(doctor_recc_h1n1),
        'doctor_recc_seasonal': int(doctor_recc_seasonal),
        'chronic_med_condition': int(chronic_med_condition),
        'child_under_6_months': 0,
        'health_worker': int(health_worker),
        'health_insurance': int(health_insurance),
        'opinion_h1n1_vacc_effective': 3,
        'opinion_h1n1_risk': 3,
        'opinion_h1n1_sick_from_vacc': 3,
        'opinion_seas_vacc_effective': 3,
        'opinion_seas_risk': 3,
        'opinion_seas_sick_from_vacc': 3,
        'household_adults': 0,
        'household_children': 0,
    }

    # Process input and make predictions
    if st.button('🔮 Generate Predictions', type='primary'):
        with st.spinner('Analyzing your profile...'):
            # Create input dataframe
            input_df = pd.DataFrame(data, index=[0])
            
            # Add dummy variables
            categorical_features = ['age_group', 'education', 'race', 'sex']
            for feature in categorical_features:
                dummies = pd.get_dummies(pd.Series([locals()[feature]]), prefix=feature)
                input_df = pd.concat([input_df, dummies], axis=1)

            # Align features with model
            model_features = h1n1_model.get_booster().feature_names
            input_aligned = pd.DataFrame(0, index=[0], columns=model_features)
            for col in input_df.columns:
                if col in model_features:
                    input_aligned[col] = input_df[col]

            # Generate predictions
            h1n1_pred = h1n1_model.predict_proba(input_aligned)[0][1]
            seasonal_pred = seasonal_model.predict_proba(input_aligned)[0][1]

        # Display results 
        gauge_col1, gauge_col2 = st.columns(2)
        with gauge_col1:
            st.plotly_chart(plot_confidence_gauge(h1n1_pred, "H1N1 Vaccine Confidence"), use_container_width=True)
        with gauge_col2:
            st.plotly_chart(plot_confidence_gauge(seasonal_pred, "Seasonal Flu Vaccine Confidence"), use_container_width=True)
        
        st.success('Analysis Complete!')
        
        results_col1, results_col2 = st.columns(2)
        
        with results_col1:
            st.markdown("### 📊 Prediction Results")
            st.metric(
                "H1N1 Vaccine Recommendation",
                f"{h1n1_pred:.1%}",
                delta="Recommended" if h1n1_pred > 0.5 else "Not Recommended"
            )
            st.metric(
                "Seasonal Flu Vaccine Recommendation",
                f"{seasonal_pred:.1%}",
                delta="Recommended" if seasonal_pred > 0.5 else "Not Recommended"
            )

        with results_col2:
            st.markdown("### 📋 Detailed Analysis")
            confidence_h1n1 = "High" if abs(h1n1_pred - 0.5) > 0.3 else "Moderate"
            confidence_seasonal = "High" if abs(seasonal_pred - 0.5) > 0.3 else "Moderate"
            
            st.markdown(f"""
            - **H1N1 Vaccine**: {'✅' if h1n1_pred > 0.5 else '❌'} 
              - Confidence: {confidence_h1n1}
            - **Seasonal Flu**: {'✅' if seasonal_pred > 0.5 else '❌'}
              - Confidence: {confidence_seasonal}
            """)

with tab2:
    st.markdown("### 🦠 About Vaccines")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown("""
        #### H1N1 Vaccine
        The H1N1 vaccine helps protect against the H1N1 flu virus. Key points:
        - Specifically targets the H1N1 strain
        - Single dose for most adults
        - Proven effective in preventing H1N1 infection
        - Minimal side effects
        """)
        
    with info_col2:
        st.markdown("""
        #### Seasonal Flu Vaccine
        The seasonal flu vaccine is updated yearly to protect against current flu strains:
        - Updated annually for new strains
        - Recommended for everyone 6 months and older
        - Reduces flu illness severity
        - Protects vulnerable populations
        """)

    st.markdown("### Why Get Vaccinated?")
    st.markdown("""
    - Protect yourself and others
    - Reduce severity of illness if infected
    - Help maintain community immunity
    - Prevent spread to vulnerable populations
    - Reduce healthcare system burden
    """)

with tab3:
    st.markdown("### 🤖 FAQ Assistant")
    st.markdown("Select a question from the dropdown below to get instant answers about vaccines.")
    
    # Define FAQ dictionary with markdown formatting
    faqs = {
        "What are the common side effects of vaccines?": """
        ### Common Side Effects
        Most side effects are mild and typically resolve within 1-2 days:

        #### Local Reactions
        * Soreness at injection site
        * Mild swelling
        * Redness

        #### General Symptoms
        * Mild fever
        * Fatigue
        * Headache

        > Note: These are signs your body is building protection
        """,
        
        "How long does vaccine immunity last?": """
        ### Immunity Duration
        
        #### H1N1 Vaccine
        * Generally 1 year of protection
        * Peak immunity after 2 weeks
        * Annual vaccination recommended

        #### Seasonal Flu Vaccine
        * Protection lasts one flu season
        * Effectiveness varies by strain match
        * Annual updates needed
        """,
        

        "Who should not get vaccinated?": """
        ### Vaccination Contraindications
        
        #### Medical Conditions
        * Severe allergic reactions to previous vaccines
        * Current moderate to severe illness
        * History of Guillain-Barré syndrome
        
        #### Special Groups
        * Infants under 6 months (for flu vaccines)
        * Pregnant women should consult healthcare provider
        * People with certain autoimmune conditions
        
        > ⚠️ Always consult your healthcare provider if unsure
        """,

        "Can I get vaccinated while pregnant?": """
        ### Vaccination During Pregnancy

        #### Benefits
        * Protects mother and developing baby
        * Passes antibodies to fetus
        * Reduces risk of complications

        #### Recommendations
        * CDC recommends flu shots during pregnancy
        * Any trimester is safe
        * Inactivated vaccines are preferred
        
        > 👶 Protecting two lives with one vaccination
        """,

        "What's the difference between flu vaccines?": """
        ### Types of Flu Vaccines

        #### Standard-Dose Flu Shot
        * For people 6 months to 64 years
        * Most common type
        * Quadrivalent protection
        
        #### High-Dose Flu Shot
        * Specifically for 65+ years
        * Contains 4x antigen amount
        * Better immune response
        
        #### Nasal Spray Vaccine
        * Age 2-49 years
        * Not for pregnant women
        * Contains weakened live viruses
        
        > 💉 Your doctor will recommend the best type for you
        """,

        "When is the best time to get vaccinated?": """
        ### Optimal Vaccination Timing

        #### General Guidelines
        * Early fall (September-October)
        * Before flu season peaks
        * At least 2 weeks before exposure
        
        #### Special Timing
        * Children need 2 doses 4 weeks apart
        * Southern hemisphere: different schedule
        * Year-round availability
        
        > 📅 Earlier vaccination provides better protection
        """,

        "How effective are these vaccines?": """
        ### Vaccine Effectiveness

        #### H1N1 Vaccine
        * 70-90% effective in healthy adults
        * Protection begins in 2 weeks
        * Lasts through flu season
        
        #### Seasonal Flu Vaccine
        * 40-60% reduction in flu risk
        * Varies by season and strain match
        * Reduces severity even if infected
        
        > 📊 Effectiveness varies by age and health status
        """
    
    }
    
    # Create selectbox for questions
    selected_question = st.selectbox(
        "Choose your question:",
        options=list(faqs.keys()),
        key="faq_select"
    )
    
    # Display answer 
    if selected_question:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style='background-color: #000000; padding: 1rem; border-radius: 10px;'>
                {faqs[selected_question]}
            """, 
            unsafe_allow_html=True
    )

        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Was this helpful?")
        
        # Create columns for feedback buttons and message
        col1, col2, col3 = st.columns([1,1,8])
        
        # Store feedback state in session state if not already present
        if 'feedback_given' not in st.session_state:
            st.session_state.feedback_given = False
            
        # Only show buttons if feedback hasn't been given
        if not st.session_state.feedback_given:
            with col1:
                if st.button("👍 Yes", key=f"yes_{selected_question}"):
                    st.session_state.feedback_given = True
                    st.session_state.feedback_message = "Thanks for your positive feedback! We're glad this was helpful."
            with col2:
                if st.button("👎 No", key=f"no_{selected_question}"):
                    st.session_state.feedback_given = True
                    st.session_state.feedback_message = "Thanks for your feedback. We'll work on improving our answers."
        
        # Show thank you message if feedback was given
        if st.session_state.feedback_given:
            st.success(st.session_state.feedback_message)

# Footer
    st.markdown("---")
    st.markdown(
    f"""
    <div style='text-align: center; color: gray;'>
        © {datetime.now().year} VacSmart | Made by Manzaar<br>
        Version 1.0.0 | 
        <a href="mailto:your@email.com" style="color: gray;">Contact</a> | 
        <a href="https://github.com/your-repo" style="color: gray;">GitHub</a><br>
        Last Updated: {datetime.now().strftime('%B %d, %Y')}
    </div>
    """, 
    unsafe_allow_html=True
    )