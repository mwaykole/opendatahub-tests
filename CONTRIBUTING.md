# Contributing to OpenDataHub Tests

Thank you for your interest in contributing to OpenDataHub Tests!

## Developer Certificate of Origin (DCO)

This project requires all commits to be signed off, indicating that you agree to the [Developer Certificate of Origin (DCO)](https://developercertificate.org/).

### How to Sign Off Your Commits

To sign off your commits, add the `-s` or `--signoff` flag to your git commit command:

```bash
git commit -s -m "Your commit message"
```

This will automatically add a `Signed-off-by` line to your commit message:

```
Signed-off-by: Your Name <your.email@example.com>
```

### Signing Off Existing Commits

If you forgot to sign off your commits, you can fix them:

**For the last commit:**
```bash
git commit --amend --signoff
git push --force-with-lease
```

**For multiple commits:**
```bash
git rebase --signoff HEAD~N  # where N is the number of commits
git push --force-with-lease
```

### Automatic DCO Check

All pull requests are automatically checked for DCO sign-off. PRs without proper sign-off will not be merged.

## Development Guidelines

Please refer to the [Developer Guide](./docs/DEVELOPER_GUIDE.md) and [Style Guide](./docs/STYLE_GUIDE.md) for detailed information about development practices and code standards.

## Getting Started

See the [Getting Started Guide](./docs/GETTING_STARTED.md) for information on setting up your development environment and running tests.

## Questions?

If you have any questions about contributing, feel free to open an issue or reach out to the maintainers.

