import os
import subprocess
import sys

from flask import Blueprint, render_template, redirect, Response

bp = Blueprint("train", __name__, url_prefix="/train")
proc: subprocess.Popen = None


def get_train_status():
    return proc is not None and proc.poll() is None


@bp.route("/")
def index():
    return render_template("train/index.html", is_training=get_train_status())


@bp.route("/start", methods=["POST"])
def start():
    global proc
    if not get_train_status():
        proc = subprocess.Popen([sys.executable, "-u", os.path.join(os.getcwd(), "trainer", "lib", "train.py")],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)

        return "Success"

    return "Training already in progress", 403


@bp.route("/stream")
def stream():
    def generate():
        global proc
        if get_train_status():
            for line in proc.stdout:
                yield "data: "
                yield str(line, encoding="utf-8")
                yield "\n\n"

    return Response(generate(), mimetype="text/event-stream")


@bp.route("/status")
def status():
    return render_template("train/status.html")
