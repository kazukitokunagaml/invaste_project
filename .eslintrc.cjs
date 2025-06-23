module.exports = {
  parser: '@typescript-eslint/parser',
  plugins: ['@typescript-eslint'],
  extends: ['eslint:recommended', 'plugin:@typescript-eslint/recommended'],
  env: {
    browser: true,
    es2020: true
  },
  rules: {
    '@typescript-eslint/no-unused-vars': 'off'
  }
};
