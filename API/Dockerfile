FROM python:3.9

WORKDIR /API

COPY ./requirements.txt /API/requirements.txt
COPY ./ /API/

RUN python -m pip install --upgrade pip
RUN python -m venv venv
ENV PATH="venv/bin:$PATH"
RUN pip install --upgrade -r /API/requirements.txt

CMD uvicorn solver_api:app --host 0.0.0.0 --port 8001 --workers 3