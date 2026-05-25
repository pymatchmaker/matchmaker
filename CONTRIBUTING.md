# Contributing

You can help the developers of **Matchmaker** by contributing, requesting features, or reporting errors. If you are interested in learning how to contribute a custom score follower, please read [this document](HOW_TO_MAKE_CUSTOM_SCORE_FOLLOWERS.md) as well.

## Opening an Issue

To open an issue, navigate to the Matchmaker GitHub repository:

[Matchmaker Issues](https://github.com/pymatchmaker/matchmaker/issues)

#### Write your description

Give as much detail as possible — for bugs, include steps to reproduce, the Python version you're using, and any relevant error messages or tracebacks.

#### Choose the appropriate label

- **Question** to ask us something, or **help wanted** if you need a solution to a particular Matchmaker problem.
- **Bug** to report something not working correctly.
- **Enhancement** to request a new feature.

---

## How to Contribute

A step-by-step guide:

1. Open a relevant issue (see above).
2. **Fork** the Matchmaker repo.
3. *Checkout* or *Pull* the latest stable `develop` branch.
4. *Checkout a new branch* from `develop`, named after your feature or fix.
5. When finished coding, open a pull request targeting the `develop` branch of Matchmaker.

### Open a Relevant Issue

Follow the section above on how to open an issue. Every contribution should have a corresponding issue — this keeps things documented and opens the floor for discussion before work begins.

### **Fork** the Repo

Fork Matchmaker from:
<https://github.com/pymatchmaker/matchmaker>

Once you have forked the repo, clone it locally:

```shell
git clone https://github.com/YourUsername/matchmaker.git
cd matchmaker
```

### Get the Latest Develop Branch

```shell
git fetch upstream
git checkout develop
git pull
```

### Create your Branch

Use a meaningful branch name that reflects what you're working on:

```shell
git checkout -b feature/your_feature_name
# or
git checkout -b bug/issue_description
```

Do your coding magic!

Remember to commit regularly with descriptive messages about your changes.

**⚠️ IMPORTANT NOTE ⚠️**

Write unit tests to check the compatibility of your changes and ensure the long-term stability of the codebase.

### Opening your Pull Request

Go to your forked Matchmaker repo and click **New Pull Request**.

Open a pull request from your new branch into the original **`develop`** branch (not `main`!) at:

<https://github.com/pymatchmaker/matchmaker>

##### Set the base to `develop` and the compare to your branch

In your PR description, reference the issue it addresses (e.g., `Closes #42`). This links the two together and helps reviewers understand the context.

When you open the PR, the Matchmaker test suite (including any unit tests you wrote) will run automatically.

If all tests pass, a member of the Matchmaker development team will review your work. Your feature will then be included in the next release, or a discussion will begin on your PR thread.

**Please avoid bundling multiple unrelated changes into a single PR.** It makes reviewing significantly harder. We'd much rather review several small, focused PRs than one large one.

---

*This contributing guide was adapted from the [Partitura](https://github.com/CPJKU/partitura) project.*
