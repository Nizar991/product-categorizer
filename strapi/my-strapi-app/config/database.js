// module.exports = ({ env }) => ({
//   connection: {
//     client: 'postgres',
//     connection: {
//       connectionString: env('DATABASE_URL'),
//       ssl: {
//         rejectUnauthorized: false, // allow self-signed certs
//       },
//     },
//     debug: false,
//   },
// });

module.exports = ({ env }) => {
  console.log('DATABASE_URL:', env('DATABASE_URL'));
  console.log('All env vars:', process.env.DATABASE_URL);
  return {
    connection: {
      client: 'postgres',
      connection: {
        connectionString: env('DATABASE_URL'),
        ssl: {
          rejectUnauthorized: false,
        },
      },
      debug: false,
    },
  };
};