import streamlit as st
import joblib
import pickle
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="OneClass SVM Anomaly Detection",
    page_icon="🔍",
    layout="wide"
)

# Load the model with joblib and pickle fallback
@st.cache_resource
def load_model():
    """Load model using joblib first, then fallback to pickle"""
    
    # Try joblib first (recommended for scikit-learn models)
    try:
        loaded_object = joblib.load('OneClass_SVM.sav')
        st.info(f"✅ Model loaded successfully using joblib")
        st.info(f"🔍 Loaded object type: {type(loaded_object)}")
        
        # Check if it's a valid model with predict method
        if hasattr(loaded_object, 'predict') and hasattr(loaded_object, 'decision_function'):
            st.success("✅ Valid model with predict and decision_function methods found!")
            return loaded_object
        elif hasattr(loaded_object, 'predict'):
            st.warning("⚠️ Model has predict method but no decision_function. Will use predict only.")
            return loaded_object
        else:
            st.error(f"❌ Loaded object doesn't have required methods. Object type: {type(loaded_object)}")
            if hasattr(loaded_object, '__dict__'):
                st.write("Available attributes:", list(loaded_object.__dict__.keys()) if hasattr(loaded_object, '__dict__') else "No __dict__")
            return None
            
    except Exception as joblib_error:
        st.warning(f"⚠️ Joblib loading failed: {str(joblib_error)}")
        st.info("🔄 Trying to load with pickle...")
        
        # Fallback to pickle
        try:
            with open('OneClass_SVM.sav', 'rb') as file:
                loaded_object = pickle.load(file)
            
            st.info(f"✅ Model loaded successfully using pickle fallback")
            st.info(f"🔍 Loaded object type: {type(loaded_object)}")
            
            # Check if it's a valid model with predict method
            if hasattr(loaded_object, 'predict') and hasattr(loaded_object, 'decision_function'):
                st.success("✅ Valid model with predict and decision_function methods found!")
                return loaded_object
            elif hasattr(loaded_object, 'predict'):
                st.warning("⚠️ Model has predict method but no decision_function. Will use predict only.")
                return loaded_object
            else:
                st.error(f"❌ Loaded object doesn't have required methods. Object type: {type(loaded_object)}")
                if hasattr(loaded_object, '__dict__'):
                    st.write("Available attributes:", list(loaded_object.__dict__.keys()) if hasattr(loaded_object, '__dict__') else "No __dict__")
                return None
                
        except FileNotFoundError:
            st.error("❌ Model file 'OneClass_SVM.sav' not found. Please ensure the file is in the same directory as this script.")
            return None
        except Exception as pickle_error:
            st.error(f"❌ Both joblib and pickle loading failed:")
            st.error(f"   - Joblib error: {str(joblib_error)}")
            st.error(f"   - Pickle error: {str(pickle_error)}")
            return None

def make_prediction_safe(model, input_data):
    """Safely make prediction with error handling"""
    try:
        # Check if model has predict method
        if not hasattr(model, 'predict'):
            raise AttributeError("Model object doesn't have 'predict' method")
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Try to get decision function if available
        decision_score = None
        if hasattr(model, 'decision_function'):
            try:
                decision_score = model.decision_function(input_data)
            except Exception as e:
                st.warning(f"⚠️ Could not get decision function: {str(e)}")
                decision_score = None
        
        return prediction, decision_score
        
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")

def debug_model_info(model):
    """Display detailed model information for debugging"""
    st.write("🔧 **Detailed Model Information:**")
    
    # Basic info
    st.write(f"- **Model Type**: {type(model)}")
    st.write(f"- **Model Module**: {type(model).__module__}")
    
    # Check for common attributes
    common_attrs = ['predict', 'decision_function', 'fit', 'transform', 'predict_proba', 'score']
    available_attrs = [attr for attr in common_attrs if hasattr(model, attr)]
    st.write(f"- **Available Methods**: {available_attrs}")
    
    # If it's a pipeline, show steps
    if hasattr(model, 'steps'):
        st.write(f"- **Pipeline Steps**: {[step[0] for step in model.steps]}")
    
    # If it has named_steps (Pipeline)
    if hasattr(model, 'named_steps'):
        st.write(f"- **Named Steps**: {list(model.named_steps.keys())}")
    
    # Show all attributes (limited to first 20)
    all_attrs = [attr for attr in dir(model) if not attr.startswith('_')][:20]
    st.write(f"- **All Attributes (first 20)**: {all_attrs}")

def main():
    st.title("🔍 OneClass SVM Anomaly Detection")
    st.markdown("---")
    st.write("This application uses a OneClass SVM model to detect anomalies in equipment operation data.")
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Cannot proceed without a valid model. Please check your model file.")
        st.info("💡 **Tips for fixing this issue:**")
        st.write("1. Ensure 'OneClass_SVM.sav' file exists in the app directory")
        st.write("2. Make sure the file contains a complete scikit-learn model/pipeline")
        st.write("3. Try re-saving your model using: `joblib.dump(model, 'OneClass_SVM.sav')`")
        st.stop()
    
    # Show model debugging info in expander
    with st.expander("🔍 Model Debug Information", expanded=False):
        debug_model_info(model)
    
    # Create input section
    st.header("📊 Input Parameters")
    
    # Define parameter ranges
    equipment_ids = [41, 34, 31, 36, 59, 14, 22, 80, 27, 47, 46, 13, 10,
                     6, 8, 17, 28, 24, 39, 32, 33, 25, 42, 11, 18, 38,
                     44, 15, 7, 110, 12, 21, 40, 45, 23, 4, 26, 30, 29,
                     106, 2, 3, 77, 115, 117, 43, 19, 16, 124, 126, 105, 154,
                     20, 9, 172, 163, 173, 174, 197, 207, 191, 209, 220, 5, 229,
                     205, 227, 225, 223, 203, 204, 196, 161, 221, 157, 238, 240, 211,
                     195, 232, 230, 226, 194, 192, 206, 231, 198, 175, 234, 250, 254,
                     276, 58, 228, 257, 252, 268, 208, 224, 201, 236, 210, 255, 199,
                     256, 237, 235, 259, 177, 253, 200, 233, 264, 275, 281, 272, 265,
                     262, 284, 286, 289, 288, 270, 279, 285, 280, 202, 271, 258, 263,
                     251, 282, 287, 267, 273, 274, 153, 222, 283, 266]
    
    shift_ids = [0, 1, 2]
    
    worker_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 19, 20, 22, 23, 25, 26, 28, 30, 32, 33, 35, 36, 37, 40, 41, 42, 44, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59, 61, 63, 64, 68, 76, 78, 80, 82, 83, 85, 88, 89, 93, 95, 96, 99, 100, 102, 104, 106, 110, 111, 114, 119, 120, 122, 123, 124, 125, 126, 128, 129, 130, 131, 132, 134, 135, 136, 138, 141, 143, 144, 145, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 158, 159, 161, 162, 163, 165, 166, 167, 169, 170, 172, 175, 176, 177, 181, 182, 184, 187, 189, 191, 193, 195, 197, 202, 203, 206, 207, 208, 209, 210, 213, 214, 215, 216, 217, 218, 219, 222, 223, 228, 232, 233, 234, 235, 238, 240, 241, 242, 243, 246, 248, 249, 251, 253, 255, 257, 258, 259, 260, 262, 263, 265, 266, 267, 269, 271, 274, 277, 281, 283, 284, 285, 287, 289, 294, 295, 297, 298, 299, 300, 301, 303, 304, 305, 306, 309, 310, 311, 312, 313, 317, 318, 319, 320, 321, 323, 327, 328, 330, 332, 333, 335, 336, 338, 339, 340, 343, 347, 348, 349, 350, 351, 352, 354, 355, 358, 361, 362, 363, 364, 365, 366, 369, 370, 371, 373, 375, 376, 379, 380, 381, 383, 386, 387, 390, 391, 394, 396, 398, 401, 403, 404, 405, 408, 410, 412, 415, 417, 418, 421, 422, 425, 431, 433, 436, 438, 439, 440, 441, 442, 443, 444, 445, 446, 448, 449, 450, 451, 453, 454, 455, 456, 461, 462, 1462, 1464, 2470, 2475, 2476, 2500, 2503, 2507, 2510, 2567, 2570, 2571, 2576, 2579, 2583, 2587, 2602, 2705, 2707, 2716, 2719, 2723, 2738, 2742, 2751, 2777, 2781, 2784, 2790, 2791, 2793, 2794, 2800, 2801, 2803, 2804, 2805, 2808, 2809, 2810, 2812, 2813, 2817, 2825, 2834, 2837, 2843, 2847, 2848, 2849, 2850, 2862, 2865, 2902, 2962, 2970, 2985, 2986, 2993, 2999, 3001, 3003, 3005, 3008, 3016, 3017, 3039, 3040, 3046, 3053, 3058, 3059, 3060, 3063, 3065, 3068, 3072, 3073, 3082, 3083, 3094, 3095, 3096, 3102, 3106, 3113, 3165, 3166, 3170, 3171, 3172, 3173, 3174, 3175, 3178, 3179, 3180, 3181, 3182, 3184, 3199, 3201, 3205, 3210, 3213, 3214, 3228, 3238, 3243, 3249, 3280, 3282, 3283, 3285, 3286, 3288, 3289, 3327, 3333, 3339, 3340, 3341, 3342, 3345, 3357, 3461, 3463, 3464, 3510, 3511, 3512, 3513, 3514, 3515, 3516, 3529, 3530, 3531, 3533, 3534, 3555, 3560, 3561, 3562, 3563, 3564, 3565, 3567, 3568, 3569, 3570, 3576, 3577, 3578, 3579, 3580, 3592, 3593, 3594, 3595, 3596, 3597, 3598, 3599, 3600, 3608, 3609, 3610, 3611, 3613, 3621, 3633, 3635, 3636, 3637, 3638, 3639, 3640, 3654, 3656, 3657, 3659, 3660, 3661, 3673, 3689, 3691, 3692, 3693, 3694, 3695, 3696, 3697, 3699, 3700, 3701, 3702, 3703, 3704, 3705, 3706, 3707, 3708, 3709, 3710, 3718, 3719, 3720, 3721, 3722, 3723, 3724, 3725, 3726, 3727, 3728, 3729, 3741, 3743, 3744, 3745, 3746, 3747, 3748, 3749, 3751, 3757, 3759, 3760, 3761, 3762, 3763, 3764, 3765, 3766, 3767, 3774, 3776, 3777, 3778, 3781, 3783, 3784, 3792, 3793, 3794, 3802, 3803, 3807, 3809, 3814, 3817, 3818, 3819, 3820, 3832, 3833, 3834, 3835, 3836, 3838, 3839, 3857, 3858, 3859, 3860, 3862, 3864, 3865, 3866, 3868, 3874, 3876, 3879, 3880, 3888, 3893, 3895, 3897, 3939, 3940]
    
    location_ids = [0, 4, 5, 6, 7, 8, 9, 17, 18, 19, 20, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 40, 41, 42, 43, 49, 56, 57, 58, 59, 64, 67, 70, 74, 86, 97, 99, 1098, 1099, 1100, 1102, 1103, 1105, 1106, 1110, 1111, 1113, 1115, 1116, 1117, 1120, 2132, 2137, 2142, 2143, 2149, 2152, 2154, 2157, 2161, 2162, 2173, 2188, 2190, 2191, 2195, 2196, 2215, 2224, 2236, 2239, 2242, 2243, 2255, 2271, 2277, 2282, 2283, 2284, 2286, 2319, 2342, 2346, 2348, 2349, 2356, 2361, 2363, 2365, 2366, 2373, 2375, 2380, 2388, 2406, 2407, 2435, 2443, 2447, 2450, 2466, 2467, 2470, 2515, 2517, 2549, 2552, 2574, 2575, 2576, 2584, 2596, 2615, 2616, 2621, 2637, 2681, 2696, 2705, 2706, 2738, 2744, 2745, 2746, 2747, 2748, 2749, 2750, 2751, 2752, 2753, 2754, 2755, 2756, 2760, 2762, 2763, 2764, 2766, 2768, 2787, 2789, 2793, 2812, 2826, 2925, 2933, 2934, 2944, 2956, 2968, 2987, 2992, 2993, 2994, 3008, 3033, 3040, 3042, 3043, 3047, 3048, 3059, 3060, 3062, 3064, 3068, 3071, 3112, 3113, 3159, 3170, 3173, 3175, 3179, 3180, 3181, 3182, 3187, 3196, 3202, 3206, 3227, 3238, 3240, 3243, 3253, 3267, 3269, 3270, 3271, 3272, 3273, 3275, 3276, 3277, 3279, 3285, 3287, 3290]
    
    activity_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
    fl_max_speeds = [15, 20, 22, 25, 30, 40, 50, 60, 70]
    
    # Create layout with columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        equipment_id = st.selectbox("🔧 Equipment ID", sorted(equipment_ids), 
                                  help="Select the equipment identifier")
        shift_id = st.selectbox("⏰ Shift ID", sorted(shift_ids),
                               help="Select the work shift (0, 1, or 2)")
    
    with col2:
        worker_id = st.selectbox("👤 Worker ID", sorted(worker_ids),
                                help="Select the worker identifier")
        location_id = st.selectbox("📍 Location ID", sorted(location_ids),
                                  help="Select the location identifier")
    
    with col3:
        activity_id = st.selectbox("🎯 Activity ID", sorted(activity_ids),
                                  help="Select the activity type")
        fl_max_speed = st.selectbox("⚡ FL Max Speed", sorted(fl_max_speeds),
                                   help="Select the maximum speed setting")
    
    st.markdown("---")
    
    # Create two columns for buttons
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    
    with col_btn2:
        predict_button = st.button("🔍 Predict Anomaly", type="primary", use_container_width=True)
    
    # Prediction logic
    if predict_button:
        # Prepare input data
        input_data = pd.DataFrame({
            'EquipmentID': [equipment_id],
            'ShiftID': [shift_id],
            'WorkerID': [worker_id],
            'LocationID': [location_id],
            'ActivityID': [activity_id],
            'FlMaxSpeed': [fl_max_speed]
        })
        
        with st.expander("📋 Input Data Debug", expanded=False):
            st.dataframe(input_data)
            st.write(f"Data shape: {input_data.shape}")
            st.write(f"Data types: {input_data.dtypes.to_dict()}")
        
        try:
            # Make prediction with error handling
            with st.spinner('Analyzing data...'):
                prediction, decision_score = make_prediction_safe(model, input_data)
            
            # Display results
            st.markdown("---")
            st.header("📋 Prediction Results")
            
            # Create columns for results
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                if prediction[0] == 1:
                    st.success("✅ **NORMAL OPERATION**")
                    st.write("The input parameters represent **normal operating conditions**.")
                else:
                    st.error("⚠️ **ANOMALY DETECTED**")
                    st.write("The input parameters may represent **unusual or anomalous conditions** that require attention.")
            
            with result_col2:
                if decision_score is not None:
                    st.metric(
                        label="Decision Score", 
                        value=f"{decision_score[0]:.4f}",
                        help="Higher positive scores indicate normal behavior, negative scores indicate anomalies"
                    )
                    
                    # Score interpretation
                    if decision_score[0] > 0.5:
                        confidence = "High confidence - Normal"
                    elif decision_score[0] > 0:
                        confidence = "Medium confidence - Normal"
                    elif decision_score[0] > -0.5:
                        confidence = "Medium confidence - Anomaly"
                    else:
                        confidence = "High confidence - Anomaly"
                    
                    st.write(f"**Confidence Level:** {confidence}")
                else:
                    st.info("Decision score not available for this model")
            
            # Display input summary
            st.subheader("📊 Input Summary")
            
            # Create a nice summary table
            summary_data = {
                'Parameter': ['Equipment ID', 'Shift ID', 'Worker ID', 'Location ID', 'Activity ID', 'FL Max Speed'],
                'Value': [equipment_id, shift_id, worker_id, location_id, activity_id, fl_max_speed]
            }
            summary_df = pd.DataFrame(summary_data)
            st.table(summary_df)
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            
            # Additional debugging information
            with st.expander("🔧 Extended Debugging Information", expanded=True):
                debug_model_info(model)
                st.write("**Input data details:**")
                st.write(f"- Input data shape: {input_data.shape}")
                st.write(f"- Input data types: {input_data.dtypes.to_dict()}")
                st.write(f"- Input data values: {input_data.values}")
                
                st.write("**Possible solutions:**")
                st.write("1. Check if the model expects different feature names")
                st.write("2. Verify the model was trained on the same feature order")
                st.write("3. Ensure the preprocessing pipeline is included in the saved model")
    
    # Information section
    st.markdown("---")
    st.header("ℹ️ About This Application")
    
    # Create expandable sections
    with st.expander("🤖 Model Information"):
        st.write("""
        This OneClass SVM (Support Vector Machine) model is specifically designed for **anomaly detection** 
        in equipment operation data. The model has been trained to learn normal operating patterns and can 
        identify potential anomalies based on the combination of input parameters.
        
        **Key Features:**
        - Pre-trained pipeline with standard scaling preprocessing
        - Optimized for industrial equipment monitoring
        - Real-time anomaly detection capabilities
        - Compatible with both joblib and pickle loading
        """)
    
    with st.expander("📊 Input Parameters Explained"):
        st.write("""
        **Equipment ID**: Unique identifier for specific equipment units in your facility
        
        **Shift ID**: Work shift classification:
        - 0: Night shift
        - 1: Day shift  
        - 2: Evening shift
        
        **Worker ID**: Unique identifier for equipment operators
        
        **Location ID**: Identifier for the physical location where equipment is operating
        
        **Activity ID**: Type of activity or operation being performed (2-12)
        
        **FL Max Speed**: Maximum operational speed setting for the equipment (15-70)
        """)
    
    with st.expander("🎯 Understanding Results"):
        st.write("""
        **Normal Operation (✅)**: The combination of input parameters represents typical, 
        expected operating conditions based on historical data patterns.
        
        **Anomaly Detected (⚠️)**: The input parameters represent unusual combinations that 
        deviate from normal patterns. This could indicate:
        - Equipment malfunction
        - Unusual operating conditions
        - Need for maintenance
        - Process optimization opportunities
        
        **Decision Score**: 
        - Positive values (>0): Normal operation (higher = more normal)
        - Negative values (<0): Potential anomaly (lower = more anomalous)
        """)
    
    with st.expander("🛠️ Troubleshooting"):
        st.write("""
        **Model Loading:**
        - App tries joblib first (recommended for scikit-learn), then falls back to pickle
        - Ensure 'OneClass_SVM.sav' file is in the same directory
        
        **Common Issues:**
        1. **Model file format**: Try re-saving your model using `joblib.dump(model, 'OneClass_SVM.sav')`
        2. **Feature mismatch**: Ensure model expects 6 features in this order: EquipmentID, ShiftID, WorkerID, LocationID, ActivityID, FlMaxSpeed
        3. **Preprocessing**: Make sure preprocessing pipeline is included in the saved model
        
        **Model Requirements:**
        - Must be a scikit-learn model with 'predict' method
        - Should include preprocessing pipeline if needed
        - Must be compatible with the input data format (6 features)
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "OneClass SVM Anomaly Detection System | Built with Streamlit<br>"
        "Model Loading: Joblib (primary) + Pickle (fallback)"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
