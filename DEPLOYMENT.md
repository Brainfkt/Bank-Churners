# Bank Churners Deployment

Bank Churners is a Streamlit application. Cloudflare Pages is used here only as a minimal static redirect layer, following the same pattern as PriceLab:

1. Keep the interactive Python app on Streamlit Community Cloud.
2. Publish `docs/` on Cloudflare Pages from this GitHub repository.
3. Attach `https://bankchurners.human---think.ing/` as the Cloudflare Pages custom domain.

## Streamlit Community Cloud

- Current app URL: `https://bank-churners-brainfkt.streamlit.app/`
- Repository: `Brainfkt/Bank-Churners`
- Branch: `main`
- Main file path: `dashboard/app.py`
- Runtime dependencies: `dashboard/requirements.txt`

## Cloudflare Pages

Recommended Pages settings:

- Project: `bank-churners`
- Source repository: `Brainfkt/Bank-Churners`
- Production branch: `main`
- Root directory: `/`
- Build command: empty
- Build output directory: `docs`
- Automatic production deployments: enabled

`docs/config.js` controls the redirect:

```js
window.BANK_CHURNERS_DEPLOYMENT = {
  streamlitAppUrl: "https://bank-churners-brainfkt.streamlit.app/",
  publicProjectUrl: "https://bankchurners.human---think.ing/"
};
```

## Custom Domain

Add the custom domain to the Cloudflare Pages project:

1. Open Cloudflare dashboard > Workers & Pages > `bank-churners`.
2. Open Custom domains.
3. Add `bankchurners.human---think.ing`.
4. Keep HTTPS enabled.

DNS for the subdomain:

```text
Type: CNAME
Name: bankchurners
Value: <YOUR-PAGES-PROJECT>.pages.dev
```

If `human---think.ing` is already managed by Cloudflare, Pages can create or validate the DNS record from the dashboard.

## Redirect Behavior

`docs/index.html` redirects directly to `streamlitAppUrl` from `docs/config.js`.

`docs/404.html` redirects unknown Cloudflare Pages paths to the same Streamlit app.
