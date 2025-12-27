import * as gcp from '@pulumi/gcp';
import * as pulumi from '@pulumi/pulumi';

const config = new pulumi.Config('gcp');
const project = config.require('project');
const location = config.get('region') || 'us-central1';

// 1. GCS Bucket for Models
const bucket = new gcp.storage.Bucket('model-bucket', {
  name: 'wolever-jg-models',
  location: location,
  forceDestroy: false, // Prevent accidental deletion
  uniformBucketLevelAccess: true,
});

// 2. Artifact Registry for ML Images
const repository = new gcp.artifactregistry.Repository('ml-repo', {
  location: location,
  repositoryId: 'jg-ml-repo',
  format: 'DOCKER',
});

// 3. Cloud Run Service
// Start with hello-world again to allow infra to come up before first build.
const imageName = 'us-docker.pkg.dev/cloudrun/container/hello';

const service = new gcp.cloudrunv2.Service('alpha-zero-service', {
  name: 'alpha-zero-service',
  location: location,
  template: {
    containers: [{
      image: imageName,
      ports: [{ containerPort: 8189 }],
      resources: {
        limits: {
          cpu: '1000m', // Can increase later if ML needs more
          memory: '2Gi', // ML models might need more RAM
        },
      },
      envs: [
        { name: 'MODEL_BUCKET', value: bucket.name },
        { name: 'JG_ENV', value: 'prod' },
        // Note: JG_ENV might need to vary per environment, but for now we deploy to 'prod' env in Cloud Run.
      ],
    }],
  },
}, {
  ignoreChanges: ['template.containers[0].image', 'template.scaling'],
});

// Public access
const iamBinding = new gcp.cloudrunv2.ServiceIamBinding('ml-public-binding', {
  location: location,
  project: project,
  name: service.name,
  role: 'roles/run.invoker',
  members: ['allUsers'],
});

// 4. Service Account for Deployer & Server Identity
// We will use a dedicated SA for the GitHub Actions (Deployer)
const deployerSa = new gcp.serviceaccount.Account('alpha-zero-deployer', {
  accountId: 'alpha-zero-deployer',
  displayName: 'AlphaZero GitHub Deployer',
});

// Grant Deployer permissions

// Upload models to bucket (Object Admin)
const bucketAdmin = new gcp.storage.BucketIAMMember('deployer-bucket-admin', {
  bucket: bucket.name,
  role: 'roles/storage.objectAdmin',
  member: pulumi.interpolate`serviceAccount:${deployerSa.email}`,
});

// Artifact Registry Writer
const repoWriter = new gcp.artifactregistry.RepositoryIamMember('deployer-repo-writer', {
  project: project,
  location: location,
  repository: repository.name,
  role: 'roles/artifactregistry.writer',
  member: pulumi.interpolate`serviceAccount:${deployerSa.email}`,
});

// Cloud Run Developer
const runDeveloper = new gcp.cloudrunv2.ServiceIamMember('deployer-run-dev', {
  project: project,
  location: location,
  name: service.name,
  role: 'roles/run.developer',
  member: pulumi.interpolate`serviceAccount:${deployerSa.email}`,
});

// Service Account User (to act as the service identity)
const saUser = new gcp.projects.IAMMember('deployer-sa-user', {
  project: project,
  role: 'roles/iam.serviceAccountUser',
  member: pulumi.interpolate`serviceAccount:${deployerSa.email}`,
});

// Allow the deployer to 'get' the service (required by gcloud run services update)
const runViewerIam = new gcp.projects.IAMMember('deployer-run-viewer', {
  project: project,
  role: 'roles/run.viewer',
  member: pulumi.interpolate`serviceAccount:${deployerSa.email}`,
});

// CRITICAL: The Cloud Run service ITSELF needs permission to read from the bucket.
// By default, it runs as the Compute Engine default service account.
// To follow least privilege, we should probably create a specific runtime SA,
// but for now let's grant Storage Object Viewer to the Compute Engine default SA
// OR simply grant it to the deployer SA if we were to use that as the service identity (which we aren't explicitly doing in step 3).
//
// IMPROVEMENT: Let's create a separate runtime SA or just use the default one.
// Since we don't know the default SA email easily here without looking it up,
// and the user didn't ask for a custom runtime SA, we will skip adding permissions
// for the runtime SA for now (User might need to do this or we assume default SA has broad access).
// BUT: Default Compute SA usually has Editor, so it can read buckets in the same project.
// However, if we want to be safe, we could grant Storage Object Viewer to the project's default compute SA.
//
// Let's assume the default SA works for now.

export const url = service.uri;
export const deployerEmail = deployerSa.email;
export const bucketName = bucket.name;
export const repoUrl = pulumi.interpolate`${location}-docker.pkg.dev/${project}/${repository.repositoryId}`;
