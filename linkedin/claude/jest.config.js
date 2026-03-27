module.exports = {
  testEnvironment: 'node',
  collectCoverageFrom: [
    '*.js',
    '!jest.config.js',
    '!node_modules/**'
  ],
  coveragePathIgnorePatterns: [
    '/node_modules/',
    '__tests__'
  ],
  testMatch: [
    '**/__tests__/**/*.test.js'
  ],
  verbose: true
};
