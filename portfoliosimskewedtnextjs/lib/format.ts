export const usd = (n: number) =>
  `$${Math.round(n).toLocaleString("en-US")}`;

export const pct = (n: number, digits = 1) => `${n.toFixed(digits)}%`;

export const pct2 = (n: number) => `${(n * 100).toFixed(2)}%`;

export const num = (n: number, digits = 2) => n.toFixed(digits);

export const parseCurrencyInput = (raw: string): number => {
  const cleaned = raw.replace(/[^\d]/g, "");
  if (!cleaned) return 0;
  return parseInt(cleaned, 10);
};

export const formatCurrencyInput = (n: number): string =>
  Math.max(0, Math.round(n)).toLocaleString("en-US");
