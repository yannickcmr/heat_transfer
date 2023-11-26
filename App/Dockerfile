FROM python:3.9

WORKDIR /App

COPY ./requirements.txt /App/requirements.txt
COPY ./ /App/

RUN python -m pip install --upgrade pip
RUN python -m venv venv
ENV PATH="venv/bin:$PATH"
RUN pip install --upgrade -r /App/requirements.txt

CMD gunicorn -b 0.0.0.0:8000 --workers 3 --threads=1 dash_app:server