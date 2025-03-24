import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import time

# Page config
st.set_page_config(
    page_title="Traffic Sign Detection",
    page_icon="🚦",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #1a1a1a, #2d2d2d);
        color: white;
    }
    .upload-box {
        border: 2px dashed #4a4a4a;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background: rgba(255,255,255,0.05);
    }
    .detection-box {
        background: rgba(255,255,255,0.1);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .confidence-meter {
        height: 20px;
        background: green;
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO("best.pt")

def create_confidence_bar(confidence):
    color = "green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"
    return f"""
        <div class="confidence-meter">
            <div style="width:{confidence*100}%; height:100%; background:{color}; transition:width 0.5s;">
            </div>
        </div>
    """

def main():
    # Header
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title("🚦 Traffic Sign Detection")
        st.markdown("### Advanced AI-Powered Traffic Sign Recognition")
    
    # Model loading with spinner
    with st.spinner("Loading AI Model..."):
        model = load_model()
    
    # File upload section
    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop your image here or click to upload",
        type=['jpg', 'jpeg', 'png']
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    if uploaded_file:
        # Image processing
        image_bytes = uploaded_file.read()
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        # Display columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Original Image")
            st.image(image, channels="BGR", use_container_width=True)
        
        # Process button
        if st.button("🔍 Analyze Image", use_container_width=True):
            with st.spinner("Analyzing image..."):
                # Add progress bar
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Inference
                image_resized = cv2.resize(image, (640, 640))
                results = model.predict(source=image_resized, conf=0.25)[0]
                plotted_image = results.plot()
                
                with col2:
                    st.markdown("### Detection Results")
                    st.image(plotted_image, channels="BGR", use_container_width=True)
                
                # Results section
                st.markdown("<div class='detection-box'>", unsafe_allow_html=True)
                st.markdown("### 📊 Detailed Analysis")
                
                if len(results.boxes) > 0:
                    for idx, box in enumerate(results.boxes):
                        class_id = box.cls.cpu().numpy()[0]
                        conf = box.conf.cpu().numpy()[0]
                        class_name = model.names[int(class_id)]
                        
                        # Create expandable section for each detection
                        with st.expander(f"Detection {idx+1}: {class_name}"):
                            st.markdown(f"**Confidence Score:** {conf:.2%}")
                            st.markdown(create_confidence_bar(conf), unsafe_allow_html=True)
                            
                            # Box coordinates
                            box_coords = box.xyxy.cpu().numpy()[0]
                            st.markdown("**Location Details:**")
                            st.code(f"X1: {box_coords[0]:.1f}, Y1: {box_coords[1]:.1f}\nX2: {box_coords[2]:.1f}, Y2: {box_coords[3]:.1f}")
                else:
                    st.warning("No traffic signs detected in this image.")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Detections", len(results.boxes))
                with col2:
                    avg_conf = np.mean(results.boxes.conf.cpu().numpy()) if len(results.boxes) > 0 else 0
                    st.metric("Average Confidence", f"{avg_conf:.2%}")
                with col3:
                    unique_classes = len(set(results.boxes.cls.cpu().numpy())) if len(results.boxes) > 0 else 0
                    st.metric("Unique Sign Types", unique_classes)

    # Footer
    st.markdown("---")
    st.markdown(
        "Made with ❤️ using YOLOv8 and Streamlit | "
        "[GitHub](https://github.com) | "
        "[Report Issue](https://github.com/issues)"
    )

if __name__ == "__main__":
    main()







# import streamlit as st
# from ultralytics import YOLO
# import cv2
# import numpy as np
# from PIL import Image
# import time
# from fpdf import FPDF
# import io
# import base64

# st.set_page_config(page_title="Traffic Sign Detection", page_icon="🚦", layout="wide")

# st.markdown("""
#     <style>
#     .stApp {
#         background: linear-gradient(to right, #1a1a1a, #2d2d2d);
#         color: white;
#     }
#     .upload-box {
#         border: 2px dashed #4a4a4a;
#         border-radius: 10px;
#         padding: 20px;
#         text-align: center;
#         background: rgba(255,255,255,0.05);
#     }
#     .detection-box {
#         background: rgba(255,255,255,0.1);
#         padding: 20px;
#         border-radius: 10px;
#         margin: 10px 0;
#     }
#     .stButton>button {
#         background-color: #FF4B4B;
#         color: white;
#         border: none;
#         padding: 10px 20px;
#         border-radius: 5px;
#         transition: background-color 0.3s;
#     }
#     .stButton>button:hover {
#         background-color: #FF2E2E;
#     }
#     </style>
# """, unsafe_allow_html=True)

# def create_pdf_report(image, results, model):
#     pdf = FPDF()
#     pdf.add_page()
    
#     # Header
#     pdf.set_font('Arial', 'B', 16)
#     pdf.cell(0, 10, 'Traffic Sign Detection Report', 0, 1, 'C')
#     pdf.ln(10)
    
#     # Save detection image
#     cv2.imwrite("temp_detection.jpg", results.plot())
    
#     # Add images
#     pdf.image("temp_detection.jpg", x=10, w=190)
#     pdf.ln(10)
    
#     # Detection details
#     pdf.set_font('Arial', 'B', 14)
#     pdf.cell(0, 10, f'Detected Objects: {len(results.boxes)}', 0, 1)
    
#     pdf.set_font('Arial', '', 12)
#     for idx, box in enumerate(results.boxes):
#         class_id = box.cls.cpu().numpy()[0]
#         conf = box.conf.cpu().numpy()[0]
#         class_name = model.names[int(class_id)]
#         pdf.cell(0, 10, f'Detection {idx+1}: {class_name} (Confidence: {conf:.2%})', 0, 1)
    
#     # Save PDF to memory
#     pdf_output = io.BytesIO()
#     pdf.output(pdf_output)
#     pdf_output.seek(0)
    
#     return pdf_output

# @st.cache_resource
# def load_model():
#     return YOLO("best.pt")

# def create_download_link(pdf_bytes):
#     b64 = base64.b64encode(pdf_bytes.read()).decode()
#     return f'<a href="data:application/pdf;base64,{b64}" download="detection_report.pdf">Download PDF Report</a>'

# def main():
#     col1, col2, col3 = st.columns([1,2,1])
#     with col2:
#         st.title("🚦 Traffic Sign Detection")
#         st.markdown("### Advanced AI-Powered Traffic Sign Recognition")
    
#     with st.spinner("Loading AI Model..."):
#         model = load_model()
    
#     st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
#     uploaded_file = st.file_uploader("Drop your image here or click to upload", type=['jpg', 'jpeg', 'png'])
#     st.markdown("</div>", unsafe_allow_html=True)
    
#     if uploaded_file:
#         image_bytes = uploaded_file.read()
#         image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.markdown("### Original Image")
#             st.image(image, channels="BGR", use_container_width=True)
        
#         if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
#             with st.spinner("Analyzing image..."):
#                 progress_bar = st.progress(0)
#                 for i in range(100):
#                     time.sleep(0.01)
#                     progress_bar.progress(i + 1)
                
#                 image_resized = cv2.resize(image, (640, 640))
#                 results = model.predict(source=image_resized, conf=0.25)[0]
#                 plotted_image = results.plot()
                
#                 with col2:
#                     st.markdown("### Detection Results")
#                     st.image(plotted_image, channels="BGR", use_container_width=True)
                
#                 # Generate and offer PDF download
#                 pdf_output = create_pdf_report(image, results, model)
#                 st.markdown(create_download_link(pdf_output), unsafe_allow_html=True)
                
#                 st.markdown("<div class='detection-box'>", unsafe_allow_html=True)
#                 st.markdown("### 📊 Detailed Analysis")
                
#                 if len(results.boxes) > 0:
#                     for idx, box in enumerate(results.boxes):
#                         class_id = box.cls.cpu().numpy()[0]
#                         conf = box.conf.cpu().numpy()[0]
#                         class_name = model.names[int(class_id)]
#                         with st.expander(f"Detection {idx+1}: {class_name}"):
#                             st.markdown(f"**Confidence Score:** {conf:.2%}")
#                             st.progress(float(conf))
#                             box_coords = box.xyxy.cpu().numpy()[0]
#                             st.code(f"X1: {box_coords[0]:.1f}, Y1: {box_coords[1]:.1f}\nX2: {box_coords[2]:.1f}, Y2: {box_coords[3]:.1f}")
#                 else:
#                     st.warning("No traffic signs detected in this image.")
                
#                 col1, col2, col3 = st.columns(3)
#                 with col1:
#                     st.metric("Total Detections", len(results.boxes))
#                 with col2:
#                     avg_conf = np.mean(results.boxes.conf.cpu().numpy()) if len(results.boxes) > 0 else 0
#                     st.metric("Average Confidence", f"{avg_conf:.2%}")
#                 with col3:
#                     unique_classes = len(set(results.boxes.cls.cpu().numpy())) if len(results.boxes) > 0 else 0
#                     st.metric("Unique Sign Types", unique_classes)

#     st.markdown("---")
#     st.markdown("Made with ❤️ using YOLOv8 and Streamlit")

# if __name__ == "__main__":
#     main()





# import streamlit as st
# from ultralytics import YOLO
# import cv2
# import numpy as np
# from PIL import Image
# import time
# import io
# import base64
# from fpdf import FPDF

# # Page config
# st.set_page_config(page_title="Traffic Sign Detection", page_icon="🚦", layout="wide")

# # Custom CSS
# st.markdown("""
#     <style>
#     .stApp {
#         background: linear-gradient(to right, #1a1a1a, #2d2d2d);
#         color: white;
#     }
#     .upload-box {
#         border: 2px dashed #4a4a4a;
#         border-radius: 10px;
#         padding: 20px;
#         text-align: center;
#         background: rgba(255,255,255,0.05);
#     }
#     .detection-box {
#         background: rgba(255,255,255,0.1);
#         padding: 20px;
#         border-radius: 10px;
#         margin: 10px 0;
#     }
#     .stButton>button {
#         background-color: #FF4B4B !important;
#         color: white !important;
#         border: none !important;
#         padding: 10px 20px !important;
#         border-radius: 5px !important;
#         transition: background-color 0.3s !important;
#     }
#     .stButton>button:hover {
#         background-color: #FF2E2E !important;
#     }
#     .download-button {
#         display: inline-block;
#         padding: 10px 20px;
#         background-color: #4CAF50;
#         color: white;
#         text-decoration: none;
#         border-radius: 5px;
#         margin: 10px 0;
#         transition: background-color 0.3s;
#     }
#     .download-button:hover {
#         background-color: #45a049;
#     }
#     .confidence-bar {
#         background-color: #4CAF50;
#         height: 20px;
#         border-radius: 10px;
#         transition: width 0.3s;
#     }
#     </style>
# """, unsafe_allow_html=True)

# def create_pdf_report(image, results, model):
#     pdf = FPDF()
#     pdf.add_page()
    
#     # Header
#     pdf.set_font('Arial', 'B', 16)
#     pdf.cell(0, 10, 'Traffic Sign Detection Report', 0, 1, 'C')
#     pdf.ln(10)
    
#     # Save detection image
#     is_success, buffer = cv2.imencode(".jpg", results.plot())
#     if is_success:
#         pdf.image(io.BytesIO(buffer), x=10, w=190)
#     pdf.ln(10)
    
#     # Detection details
#     pdf.set_font('Arial', 'B', 14)
#     pdf.cell(0, 10, f'Detected Objects: {len(results.boxes)}', 0, 1)
    
#     # Summary statistics
#     pdf.set_font('Arial', 'B', 12)
#     pdf.cell(0, 10, 'Summary Statistics:', 0, 1)
#     pdf.set_font('Arial', '', 12)
    
#     avg_conf = np.mean(results.boxes.conf.cpu().numpy()) if len(results.boxes) > 0 else 0
#     unique_classes = len(set(results.boxes.cls.cpu().numpy())) if len(results.boxes) > 0 else 0
    
#     pdf.cell(0, 10, f'Average Confidence: {avg_conf:.2%}', 0, 1)
#     pdf.cell(0, 10, f'Unique Sign Types: {unique_classes}', 0, 1)
#     pdf.ln(10)
    
#     # Detailed detections
#     pdf.set_font('Arial', 'B', 12)
#     pdf.cell(0, 10, 'Detailed Detections:', 0, 1)
#     pdf.set_font('Arial', '', 12)
    
#     for idx, box in enumerate(results.boxes):
#         class_id = box.cls.cpu().numpy()[0]
#         conf = box.conf.cpu().numpy()[0]
#         class_name = model.names[int(class_id)]
#         box_coords = box.xyxy.cpu().numpy()[0]
        
#         pdf.cell(0, 10, f'Detection {idx+1}: {class_name}', 0, 1)
#         pdf.cell(0, 10, f'Confidence: {conf:.2%}', 0, 1)
#         pdf.cell(0, 10, f'Location: X1={box_coords[0]:.1f}, Y1={box_coords[1]:.1f}, X2={box_coords[2]:.1f}, Y2={box_coords[3]:.1f}', 0, 1)
#         pdf.ln(5)
    
#     return pdf.output(dest='S').encode('latin1')

# def create_download_link(pdf_bytes):
#     b64 = base64.b64encode(pdf_bytes).decode()
#     return f'<a href="data:application/pdf;base64,{b64}" download="detection_report.pdf" class="download-button">📄 Download PDF Report</a>'

# @st.cache_resource
# def load_model():
#     return YOLO("best.pt")

# def main():
#     # Header
#     col1, col2, col3 = st.columns([1,2,1])
#     with col2:
#         st.title("🚦 Traffic Sign Detection")
#         st.markdown("### Advanced AI-Powered Traffic Sign Recognition")
    
#     # Load model
#     with st.spinner("Loading AI Model..."):
#         model = load_model()
    
#     # Upload section
#     st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
#     uploaded_file = st.file_uploader("Drop your image here or click to upload", type=['jpg', 'jpeg', 'png'])
#     st.markdown("</div>", unsafe_allow_html=True)
    
#     if uploaded_file:
#         # Process image
#         image_bytes = uploaded_file.read()
#         image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
#         # Display columns
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.markdown("### Original Image")
#             st.image(image, channels="BGR", use_container_width=True)
        
#         # Analyze button
#         if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
#             with st.spinner("Analyzing image..."):
#                 # Progress animation
#                 progress_bar = st.progress(0)
#                 for i in range(100):
#                     time.sleep(0.01)
#                     progress_bar.progress(i + 1)
                
#                 # Run detection
#                 image_resized = cv2.resize(image, (640, 640))
#                 results = model.predict(source=image_resized, conf=0.25)[0]
#                 plotted_image = results.plot()
                
#                 # Show results
#                 with col2:
#                     st.markdown("### Detection Results")
#                     st.image(plotted_image, channels="BGR", use_container_width=True)
                
#                 # Generate and display PDF download
#                 pdf_bytes = create_pdf_report(image, results, model)
#                 st.markdown(create_download_link(pdf_bytes), unsafe_allow_html=True)
                
#                 # Detailed analysis
#                 st.markdown("<div class='detection-box'>", unsafe_allow_html=True)
#                 st.markdown("### 📊 Detailed Analysis")
                
#                 if len(results.boxes) > 0:
#                     for idx, box in enumerate(results.boxes):
#                         class_id = box.cls.cpu().numpy()[0]
#                         conf = box.conf.cpu().numpy()[0]
#                         class_name = model.names[int(class_id)]
                        
#                         with st.expander(f"Detection {idx+1}: {class_name}"):
#                             st.markdown(f"**Confidence Score:** {conf:.2%}")
#                             st.progress(float(conf))
                            
#                             box_coords = box.xyxy.cpu().numpy()[0]
#                             st.code(f"""
# Location Details:
# X1: {box_coords[0]:.1f}, Y1: {box_coords[1]:.1f}
# X2: {box_coords[2]:.1f}, Y2: {box_coords[3]:.1f}
# """)
#                 else:
#                     st.warning("No traffic signs detected in this image.")
                
#                 # Summary metrics
#                 col1, col2, col3 = st.columns(3)
#                 with col1:
#                     st.metric("Total Detections", len(results.boxes))
#                 with col2:
#                     avg_conf = np.mean(results.boxes.conf.cpu().numpy()) if len(results.boxes) > 0 else 0
#                     st.metric("Average Confidence", f"{avg_conf:.2%}")
#                 with col3:
#                     unique_classes = len(set(results.boxes.cls.cpu().numpy())) if len(results.boxes) > 0 else 0
#                     st.metric("Unique Sign Types", unique_classes)

#     # Footer
#     st.markdown("---")
#     st.markdown("Made with ❤️ using YOLOv8 and Streamlit")

# if __name__ == "__main__":
#     main()