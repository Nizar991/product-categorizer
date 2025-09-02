# Use Node 18
FROM node:18

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the project
COPY . .

# Set environment variable to ignore TLS issues (like your local dev)
ENV NODE_TLS_REJECT_UNAUTHORIZED=0

# Run Strapi in development mode
CMD ["npm", "run", "develop"]
