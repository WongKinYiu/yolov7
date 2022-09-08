FROM 485240561355.dkr.ecr.us-west-2.amazonaws.com/hivemapper-ml:latest

# This would make more sense in the ML Layer but we're not rebuilding it
# automatically right now so it's getting installed here
# TODO please update when appropriate
RUN pip3 install mlflow==1.27

RUN mkdir -p /usr/src/python
WORKDIR /usr/src/python
COPY ./python/. .

# Create server app directory 
RUN mkdir -p /usr/src/server
WORKDIR /usr/src/server

# Install app dependencies
# A wildcard is used to ensure both package.json AND package-lock.json are copied
# where available (npm@5+)
COPY package*.json ./

RUN npm install
# RUN npm i phantomjs-prebuilt --unsafe-perm
# If you are building your code for production
# RUN npm ci --only=production

# Bundle app source
# This assumes a fresh git clone, like from CI/CD
COPY . .

ENTRYPOINT [ "npm", "run" ]

