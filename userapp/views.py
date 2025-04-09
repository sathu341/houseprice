from django.shortcuts import render,redirect
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
from django.contrib import messages
# Create your views here.
def home(request):
    if request.method=="POST":
        area_value=int(request.POST.get("area")) #int()
        bedroom=int(request.POST.get('bedroom'))
        bathroom=int(request.POST.get('bathroom'))
        parking=int(request.POST.get('parking'))
        stories=int(request.POST.get('stories'))
        furnishingstatus=int(request.POST.get('furnishingstatus'))
        print(area_value,bedroom,bathroom,parking,stories,furnishingstatus)
        model = joblib.load("flat_price_classifier.pkl")
        sample_data = pd.DataFrame([{
    'area':area_value,
    'bedrooms':bedroom,
    'bathrooms':bathroom,
    'stories':stories,
    'parking':parking,
    'furnishingstatus':furnishingstatus  # Furnished (assuming 1 = furnished)
}])
        prediction = model.predict(sample_data)
        message=""
        if prediction[0]==1:
            message="Prediciton:Expensive"
            
        else:
            message="Prediction:Affordable"
        messages.info(request,message)
        return redirect("/home") 
        #return render(request,"result.html",)    
            
        print("Prediction:", "Expensive" if prediction[0] == 1 else "Affordable") 
   


    return render(request,"homepage.html")

def showResult(request):
    return render(request,"result.html")