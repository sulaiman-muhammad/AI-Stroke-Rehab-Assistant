from flask import Flask, render_template, request
from exercises import arm_raise, knee_extension, sit_to_stand, pattern_tracing

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/exercise', methods=['POST'])
def exercise():
    exercise_name = request.form['exercise']
    if exercise_name == 'arm_raise':
        count, time_taken, feedback = arm_raise.run_arm_raise()
        display_name = 'Arm Raise (Upper Body)'
    elif exercise_name == 'knee_extension':
        count, time_taken, feedback = knee_extension.run_knee_extension()
        display_name = 'Knee Extension (Lower Body)'
    elif exercise_name == 'sit_to_stand':
        count, time_taken, feedback = sit_to_stand.run_sit_to_stand()
        display_name = 'Sit to Stand (Full Body)'
    elif exercise_name == 'pattern_tracing':
        count, time_taken, feedback = pattern_tracing.run_pattern_tracing()
        display_name = 'Pattern Tracing (Balance and Coordination)'
    return render_template('result.html', exercise_name=display_name, count=count, time_taken=time_taken, feedback=feedback)

if __name__ == '__main__':
    app.run(debug=True)
