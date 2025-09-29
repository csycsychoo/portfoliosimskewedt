/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  webpack: (config) => {
    config.resolve.alias = {
      ...config.resolve.alias,
      'plotly.js': 'plotly.js/dist/plotly.js',
    };
    return config;
  },
}

module.exports = nextConfig