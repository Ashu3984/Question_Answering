from transformers import pipeline
import gradio as gr

question_answerer = pipeline("question-answering")


def main(context, question):
    answer = question_answerer(
        question=question,
        context=context,
    )
    return answer


demo = gr.Interface(
    fn=main,
    inputs=["text", "text"],
    outputs="text",
    title="Question Answering"
)

demo.launch(
    inbrowser=True,
    show_error=True,
    show_tips=True,
    show_api=False)