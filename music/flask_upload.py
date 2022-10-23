from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
from pytube import YouTube
import glob
import os.path
from music21 import *
import shutil
import os

# omnizart 패키지 사용
from omnizart.music import app as mapp
from omnizart.chord import app as capp
from omnizart.drum import app as dapp
from omnizart.vocal import app as vapp
from omnizart.vocal_contour import app as vcapp
from omnizart.beat import app as bapp

from bs4 import BeautifulSoup
import requests
import urllib.request

app = Flask(__name__)


# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 #파일 업로드 용량 제한 단위:바이트


# HTML 렌더링
@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/start')
def start():
    return render_template('start2.html')


# 파일 리스트
@app.route('/list')
def list_page():
    file_list = os.listdir("./uploads")
    html = """<center><a href="/">홈페이지</a><br><br>"""
    html += "다운로드 받은 파일 목록 : {}".format(file_list) + "</center>"

    return html


# 업로드 HTML 렌더링
@app.route('/upload')
def upload_page():
    return render_template('upload.html')


# 파일 업로드 처리
@app.route('/fileUpload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save('./uploads/' + secure_filename(f.filename))
        us = environment.UserSettings()

        us['lilypondPath'] = '/home/01_File_Upload/LilyPond/usr/bin/lilypond.exe'
        us['musescoreDirectPNGPath'] = '/home/01_File_Upload/MuseScore 2/bin/MuseScore.exe'
        us['musicxmlPath'] = '/home/01_File_Upload/MuseScore 2/bin/MuseScore.exe'

        files = glob.glob("uploads/*.mid")
        for x in files:
            if not os.path.isdir(x):
                filename = os.path.splitext(x)
                try:
                    original_score = converter.parse(x).chordify()
                    conv = converter.subConverters.ConverterLilypond()
                    conv.write(original_score, fmt='lilypond', fp='score', subformats=['png'])
                    # original_score.show()
                except:
                    pass
                    '''
        source='/root/01_File_Upload/score.png'
        destination='/root/01_File_Upload/score.png'
        shutil.move(source,destination)
     '''

        return render_template('check_midi_download.html')


# url 처리
@app.route('/urlupload', methods=['GET', 'POST'])
def url_check():
    if request.method == 'POST':
        DOWNLOAD_FOLDER = "./"
        a = request.form['url']

        yt = YouTube(a)
        stream = yt.streams.get_highest_resolution()
        stream.download(DOWNLOAD_FOLDER)

        files = glob.glob("./*.mp4")
        for x in files:
            if not os.path.isdir(x):
                filename = os.path.splitext(x)
                try:
                    os.rename(x, filename[0] + '.wav')
                except:
                    pass

    trasncription_modes = ["music-piano", "music-piano-v2", "music-assemble", "chord", "drum", "vocal", "vocal-contour",
                           "beat"]

    files = glob.glob("./*.wav")
    for x in files:
        filename = os.path.splitext(x)
        file_nm = filename[0][2:]

    uploaded_audio = file_nm

    # 악보 비교 크롤링
    url_new = uploaded_audio.replace(' ', '+')
    url = "https://www.akbobada.com/searchAll.html?searchKeyword=%s&searchFlag=10&searchSecod=&viewFlag=null&artistOrderBy=null&searchSelect=null&searchSelectName=null" % url_new

    response = requests.get(url)

    if response.status_code == 200:
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        a = soup.find('div').find('a', {'class': 'tit'})['href']
        score_url = 'https://www.akbobada.com%s' % a

    new_url = score_url

    htmlcontent1 = urllib.request.urlopen(new_url).read()
    v = BeautifulSoup(htmlcontent1, 'html.parser')
    sample = v.find("div", "img_view").find("img", recursive=False)
    sample_url = sample["src"]
    download_url = sample_url
    urllib.request.urlretrieve(download_url, 'sam122.png')
    source = './sam122.png'
    destination = './static/images/sam122.png'
    shutil.move(source, destination)

    mode = "chord"
    model = ""
    if mode.startswith("music"):
        mode_list = mode.split("-")
        mode = mode_list[0]
        model = "-".join(mode_list[1:])

    app = {
        "music": mapp,
        "chord": capp,
        "drum": dapp,
        "vocal": vapp,
        "vocal-contour": vcapp,
        "beat": bapp
    }[mode]

    model_path = {
        "piano": "Piano",
        "piano-v2": "PianoV2",
        "assemble": "Stream",
        "pop-song": "Pop",
        "": None
    }[model]

    midi = app.transcribe(f"{uploaded_audio}.wav", model_path=model_path)
    source = './%s.mid' % uploaded_audio
    destination = './uploads/%s.mid' % uploaded_audio
    shutil.move(source, destination)
    remove = './%s.wav' % uploaded_audio
    os.remove(remove)

    source = './%s' % uploaded_audio
    destination = './playlist/%s' % uploaded_audio
    shutil.move(source, destination)

    # play_list = os.listdir("./uploads/playlist")

    return render_template('score.html', scurl=score_url, play_url=destination)


#    return render_template('check_url_download.html')


# url page
@app.route('/url', methods=['GET', 'POST'])
def url_page():
    return render_template('contact.html')


@app.route('/list_down', methods=['GET', 'POST'])
def list_down():
    file_list = os.listdir("./uploads")
    return render_template('down_list.html', filedown_list=file_list)


@app.route('/play', methods=['GET', 'POST'])
def play():
    play_list = os.listdir("./playlist")
    return render_template('play.html', play=play_list)


'''
@app.route('/Downnow')
def Downnow():
    file_name = f"static/results/file_path.csv"
    return send_file(file_name,
                     mimetype='text/csv',
                     attachment_filename='downloaded_file_name.csv',# 다운받아지는 파일 이름. 
                     as_attachment=True)

'''


@app.route('/midiDown', methods=['GET', 'POST'])
def midiDown():
    files = glob.glob("uploads/*.png")
    return send_file(files, attachment_filename='Music_Score.png', as_attachment=True)


# 파일 다운로드 처리
@app.route('/fileDown', methods=['GET', 'POST'])
def down_file():
    if request.method == 'POST':
        sw = 0
        files = os.listdir("./uploads")
        for x in files:
            if (x == request.form['file']):
                sw = 1
                path = "./uploads/"

                return send_file(path + request.form['file'],
                                 attachment_filename=request.form['file'],
                                 as_attachment=True)

        return render_template('page_not_found.html')
    else:
        return render_template('page_not_found.html')


@app.route('/csvdown', methods=['GET', 'POST'])
def csvdown():
    if request.method == 'POST':
        sw = 0
        files = os.listdir("./playlist")
        for x in files:
            if (x == request.form['csv']):
                sw = 1
                path = "./playlist/"

                return send_file(path + request.form['csv'],
                                 attachment_filename=request.form['csv'],
                                 as_attachment=True)

        return render_template('page_not_found.html')
    else:
        return render_template('page_not_found.html')


# 다운로드 HTML 렌더링
@app.route('/downfile')
def down_page():
    files = os.listdir("./uploads")
    return render_template('download_check.html', files=files)


if __name__ == '__main__':
    app.run(port=5000, debug = True)

