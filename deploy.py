import gradio as gr
import numpy as np
import pickle

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

def predict_survival(passenger_characteristics):
    passenger_characteristics = np.array([passenger_characteristics])

    prediction = model.predict(passenger_characteristics)

    return prediction[0][1]

# Interface
interface = gr.Interface(
  fn=predict_survival,
  inputs=[
    gr.inputs.Number(label='Pclass'),
    gr.inputs.Select(label='Sex', choices=['male', 'female']),
    gr.inputs.Number(label='Age'),
    gr.inputs.Number(label='SibSp'),
    gr.inputs.Number(label='Parch'),
    gr.inputs.Number(label='Fare'),
    gr.inputs.Select(label='Embarked', choices=['S', 'C', 'Q']),
  ],
  outputs=[gr.outputs.Label(label='Predicted survival probability')],
)

interface.launch()