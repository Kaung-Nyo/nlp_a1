FROM python:3.10.12

WORKDIR /root/code
# WORKDIR /home/knl/DSAI/ML/Car/Car_Price_Prediction_Model/code

RUN pip3 install dash
RUN pip3 install pandas
RUN pip3 install dash_bootstrap_components
RUN pip3 install dash-bootstrap-components[pandas]
RUN pip3 install pandas
RUN pip3 install numpy
RUN pip3 install scikit-learn
RUN pip3 install gensim
RUN pip3 install nltk



# Testing module
RUN pip3 install dash[testing]

COPY ./app /root/app
# COPY . ./
# CMD tail -f /dev/null

CMD python3 main.py