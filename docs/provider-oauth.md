# Browser sign-in (OAuth) for GPU providers: research notes

Status: research only, nothing implemented. Read from each provider's public docs on 2026-10-15.
No app was registered, no account created, no sign-in attempted. Today `oneapi-agent` takes a
provider API key from an environment variable or from `oneapi-agent login <provider>` (OS
password store); see [gpu-testing.md](gpu-testing.md).

`remote-test` needs a credential that can: read the account, list/price GPUs, create a
machine, read its state, delete it, and (DigitalOcean only) add and remove a throwaway SSH
key. A sign-in flow is only worth building if it can grant that and nothing more.

## Summary

| Provider | OAuth / browser sign-in for third-party CLIs | Device flow | Fine-grained access | Verdict |
|---|---|---|---|---|
| DigitalOcean | **Yes.** OAuth 2.0 authorization code + PKCE for public clients, Dynamic Client Registration, loopback redirects on any port | Not documented | Yes: token scopes such as `droplet:create` | Feasible now |
| Vultr | **Yes, but heavier.** OAuth 2.0 / OIDC apps, PKCE, consent screen, app review | Not documented | Yes, through an IAM policy attached to the app | Feasible after Vultr approves an app; several gaps in the docs |
| Runpod | **No public flow.** Runpod's own `flash login` opens the console and saves a key, but no third-party API is documented | No | API keys: All / Restricted / Read Only | Not possible with public docs; keep API keys |
| Vast.ai | **No.** API keys only | No | Scoped API keys (permissions JSON) | Not possible; keep API keys, recommend a scoped key |

## DigitalOcean

Docs: <https://docs.digitalocean.com/reference/api/oauth/>, scopes:
<https://docs.digitalocean.com/reference/api/scopes/>, user tutorial:
<https://www.digitalocean.com/community/tutorials/how-to-use-oauth-authentication-with-digitalocean-as-a-user-or-developer>.

- **Flow.** Authorization code. Public clients (native apps, CLIs) must use PKCE with `S256`;
  they get no client secret. Endpoints:
  - authorize `https://cloud.digitalocean.com/v1/oauth/authorize`
  - token / refresh `https://cloud.digitalocean.com/v1/oauth/token`
  - revoke `https://cloud.digitalocean.com/v1/oauth/revoke`
  - metadata `https://cloud.digitalocean.com/.well-known/oauth-authorization-server`
- **Registering an app.** Either in the control panel (Apps & API, Developer Applications:
  name, homepage, description, callback URL; gives a client id and secret), or, for a public
  client, with Dynamic Client Registration: an unauthenticated `POST
  https://cloud.digitalocean.com/v1/oauth/register` that returns a client id (no secret) and a
  `registration_access_token` for managing it. One to five redirect URIs; loopback redirects
  (`127.0.0.1`, `[::1]`, `localhost`) match on any port, so a CLI can listen on an ephemeral
  local port.
- **Tokens.** Public-client access tokens last one hour; refresh tokens are single use and
  rotate. A CLI would store the refresh token in the OS password store (as `login` does today).
- **Scopes `remote-test` needs:** `account:read`, `droplet:read`, `droplet:create`,
  `droplet:delete`, `ssh_key:create`, `ssh_key:delete` (plus `ssh_key:read` if listing keys).
  Without `scope`, the consent screen lets the user pick granular scopes.
- **Unclear:** the OAuth page does not list the scope names itself; the scopes page is titled
  "Scopes for API Tokens" and does not say outright that every name is valid in an OAuth
  `scope` parameter. GPU droplet sizes may need extra scopes (e.g. `regions:read`,
  `sizes:read`); not checked. No device flow is documented, so a headless machine (CI, SSH
  box) cannot use this; it would keep using `OCEAN_API_KEY`.

## Vultr

Docs: <https://docs.vultr.com/platform/iam/oauth>,
<https://docs.vultr.com/platform/iam/oauth/access-tokens/how-to-integrate-a-third-party-app-with-vultr-oauth>,
<https://docs.vultr.com/platform/iam/oauth/access-tokens/how-to-use-pkce-with-vultr-oauth>,
<https://docs.vultr.com/platform/iam/oauth/authorizations/how-to-authorize-a-third-party-oauth-application>.

- **Flow.** Vultr acts as an OAuth 2.0 / OIDC provider. The app owner creates an OIDC provider
  (`POST /v2/oidc/provider`), registers a client (`POST /v2/oauth/clients`, with callback URLs;
  the client secret is shown once) and attaches a permission set (`POST
  /v2/oauth/clients/{id}/scopes` with an IAM policy id). Customers consent in the Vultr
  console; the code (single use, 10 minutes) is exchanged at `POST
  https://api.vultr.com/v2/oidc/provider/{provider-id}/token`. The access token is a JWT sent
  as `Bearer`, like an API key. Vultr creates a dedicated role limited to what the customer
  approved, and the customer can revoke it later.
- **Registering an app.** App management needs the organisation root user's API key.
  Publishing needs a funded account, at least one `https` callback URL, at least one scope,
  and Vultr's approval; until then only a draft-mode allowlist can authorize it.
- **Scopes.** Not named strings: the app is given an IAM policy. `remote-test` needs instance
  create/read/delete, plan and region reads, and account read; the exact IAM action names were
  not checked.
- **Tokens.** Refresh tokens rotate; reusing an old one revokes the whole family; they expire
  after 30 days without use.
- **Unclear:** whether a public client can skip the client secret (the PKCE article's token
  example still sends Basic auth with the secret), whether loopback/`http://localhost`
  callbacks are accepted (only `https` is stated as a publishing requirement), the access
  token lifetime, and any device flow (none documented). The authorize call is documented as
  an API `POST`; the console consent URL a CLI would open in a browser is not given.

## Runpod

Docs: <https://docs.runpod.io/get-started/credentials>, <https://docs.runpod.io/flash/cli/login>,
<https://docs.runpod.io/runpodctl/reference/runpodctl-config>.

- **No documented OAuth for third parties.** Runpod's own Flash CLI has `flash login`: it opens
  "the Runpod console authorization page", waits for approval (default 600 s; `--no-open`
  prints the URL) and saves an API key to `~/.runpod/config.toml`. The docs do not say whether
  this is OAuth, what endpoints it uses, or whether other apps may use it. Using an
  undocumented flow is not an option.
- `runpodctl` itself only takes an API key (`runpodctl config --apiKey`, or `runpodctl doctor`).
- **Keys.** Created at <https://console.runpod.io/user/credentials> (API Keys tab) with
  permission All, Restricted (per Serverless endpoint) or Read Only. Pods need full (All)
  access as far as the docs show; Restricted is described for Serverless endpoints only.
- **Unclear:** whether a Restricted key can be limited to pods; whether Runpod would offer the
  `flash login` flow to third parties (would need to ask Runpod).

## Vast.ai

Docs: <https://docs.vast.ai/cli/authentication>, <https://docs.vast.ai/guides/reference/api-keys.md>,
<https://docs.vast.ai/api-reference/authentication.md>.

- **No OAuth or browser sign-in** is documented. The official CLI uses `vastai set api-key`
  (stored in `~/.config/vastai/vast_api_key`) or `VAST_API_KEY`.
- **Scoped keys** exist: created at <https://cloud.vast.ai/manage-keys/> or with `vastai create
  api-key --permissions '{...}'`. A key with instance management but no billing or key
  management would be the least-privilege choice for `remote-test`.
- **Unclear:** the exact permission flags `remote-test` needs (offer search, rent, show,
  destroy, `users/current` for `providers`); the permissions reference
  (<https://docs.vast.ai/api-reference/permissions>) was not read in detail.

## Recommendation

1. Keep API keys plus `oneapi-agent login` as the default for all four providers, and
   recommend least-privilege keys in the docs (Vast scoped key, Runpod where possible).
2. If browser sign-in is wanted, start with **DigitalOcean**: PKCE public client, Dynamic
   Client Registration (or one registered app), loopback redirect, the six scopes above,
   refresh token in the OS password store. CI keeps using `OCEAN_API_KEY`.
3. Vultr second, after confirming with Vultr that a public client with a loopback callback
   is allowed, and after an app review.
4. Runpod and Vast.ai: nothing to build until they publish a third-party flow.
