
# **Check Extraction & Analytics System**  

A **Streamlit-based web application** for **automated check extraction** from PDFs, **data visualization**, and **Excel export functionality**.  

## **🔹 Project Features**  

✅ **User Authentication**  
- Users can **register** and **log in** to access the system.  
- Authentication is **session-based** (stored in-memory for now).  

✅ **PDF Upload & Processing**  
- Users can upload a **PDF containing checks**.  
- The system **extracts all images** from the PDF.  
- Simulated **OCR processing** of extracted checks (currently uses **dummy data**).  

✅ **Data Display & Export**  
- Extracted check data is shown in **a table**.  
- Users can **download the data** as an **Excel file**.  

✅ **Data Analytics**  
- **Total number of checks** and **total transaction amount**.  
- **Bar chart visualization** of extracted check amounts.  

---

## **🔹 Tech Stack Used**  

🔹 **Frontend & Backend**: [Streamlit](https://streamlit.io/)  
🔹 **Data Handling**: Pandas, OpenPyXL  
🔹 **File Processing**: pdf2image (for extracting images from PDF)  
🔹 **Visualization**: Streamlit Charts  

---

## **🔹 What I Learned**  

✅ **User authentication in Streamlit** (session state management).  
✅ **Handling file uploads** and processing PDFs in Python.  
✅ **Extracting and displaying data dynamically** in a web application.  
✅ **Data visualization techniques** (bar charts, analytics).  
✅ **Generating downloadable Excel files** using Pandas and OpenPyXL.  

---

## **🔹 How to Run the Project**  

### **1️⃣ Install Dependencies**  
Run the following command to install required packages:  

```bash
pip install streamlit pandas openpyxl pdf2image
```

### **2️⃣ Run the Application**  
Execute the following command to start the app:  

```bash
streamlit run main.py
```

### **3️⃣ Open in Browser**  
After running the command, **open the displayed URL** (e.g., `http://localhost:8501/`) in your browser.

---

## **🔹 Screenshots & Output**  

### **📌 1. Login Page**  
🔹 Users must log in to access the system.  
![Login Page](Output_Images_Gui\LoginPAge.jpeg)  

---

### **📌 2. Register Page**  
🔹 New users can create an account.  
![Register Page](Output_Images_Gui\Registerpage.jpeg)  

---

### **📌 3. Upload PDF & Process Checks**  
🔹 Users can upload a **PDF file** for check extraction.  
![Upload PDF](Output_Images_Gui\Check_Upload_Process_page.jpeg)  

---

### **📌 4. Extracted Check Data (Dummy Data)**  
🔹 Extracted check data is shown in a **table** and **can be downloaded as Excel**.  
![Extracted Data](Output_Images_Gui\Check_Upload_Process_page.jpeg)  

---

### **📌 5. Data Analytics & Visualization**  
🔹 Shows **total checks**, **total transaction amount**, and **bar chart of amounts**.  
![Analytics](Output_Images_Gui\Analytics_page4.jpeg)  
![Analytics](Output_Images_Gui\Analytics_page2.jpeg)  
![Analytics](Output_Images_Gui\Analytics_page3.jpeg)  
![Analytics](Output_Images_Gui\Analytics_page1.jpeg)  

---

## **🔹 Future Improvements**  

🚀 **Implement actual OCR-based text extraction** (e.g., with Tesseract).  
🚀 **Store user data in a database** (SQLite, PostgreSQL, Firebase).  
🚀 **Enhance security** (hash passwords, proper authentication flow).  
🚀 **Improve UI/UX** with better styles and animations.  

---

## **💡 Contributing**  
Have suggestions or improvements? Feel free to fork and contribute! 😊  

---

## **📜 License**  
This project is **open-source** and available under the **MIT License**.  

---

Let me know if you need any modifications! 🚀